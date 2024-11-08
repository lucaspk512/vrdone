import argparse
import os
import yaml
import time
import json
import atexit
from functools import partial

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import utils.misc as utils
from utils.logging import setup_logger
from utils.train_utils import ModelEma, build_optimizer, build_scheduler, save_checkpoint
from dataloaders import VidVRD, VidOR
from models.maskvrd import MaskVRD

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Video Relation Detector")

    # control
    parser.add_argument("--data_name", type=str, choices=['vidor', 'vidvrd'], help="dataset name")
    parser.add_argument("--cfg_path", type=str, help="configuration file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--exp_dir", type=str, help="experiment path to save logs and ckpts")
    parser.add_argument("--from_checkpoint", action="store_true", default=False, help="if resume")
    parser.add_argument("--ckpt_path", type=str, help="ckpt path to resume") 
    parser.add_argument("--scale", default=None, type=int)
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ## load configs
    with open(args.cfg_path, 'r', encoding='utf-8') as cf:
        cfg_file = cf.read()
    config = yaml.safe_load(cfg_file)

    ## update config
    config['training_config']['seed'] = args.seed
    config['model_config']['with_clip_feature'] = config['dataset_config'].get('with_clip_feature', False)
    config['dataset_config'].update(config['training_dataset_config'])

    ## DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    ## set random seed, create work_dir, and save config
    utils.set_seed(args.seed, args.disable_deterministic)
    if args.rank == 0:
        utils.create_folder(args.exp_dir)
        utils.create_folder(os.path.join(args.exp_dir, 'logfile'))
        utils.save_config(config, args.exp_dir)

    ## setup logger
    logger = setup_logger("Train", save_dir=os.path.join(args.exp_dir, 'logfile'), distributed_rank=args.rank, filename="train_log.json")
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{json.dumps(config, indent=4)}")

    ## construct data
    if args.data_name == 'vidor':
        dataset = VidOR(config['dataset_config'], args.scale)
    else:
        dataset = VidVRD(config['dataset_config'])

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        drop_last=True,
    )

    num_workers = config['training_config']["num_workers"]
    batch_size = config['training_config']["batch_size"]
    assert batch_size % args.world_size == 0, f"batch size {batch_size} should be divided by world size {args.world_size}"
    logger.info(f"Batch size: {batch_size * config['training_dataset_config']['num_pairs']}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['training_config']["batch_size"] // args.world_size,
        collate_fn=dataset.collator_func,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    ## construct model
    device = f"cuda:{args.local_rank}"
    model = MaskVRD(config['model_config'], device=device)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False, # TODO：if changed to `True`?
    )

    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    total_num = sum([p.numel() for p in model.parameters()])
    trainable_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info("Number of model.parameters: total:{}, trainable:{}".format(total_num, trainable_num))
    
    logger.info("Using Model EMA...")
    model_ema = ModelEma(model)

    ## build optimizer and scheduler
    optimizer = build_optimizer(model, config['training_config'])
    scheduler = build_scheduler(optimizer, config['training_config'], len(dataloader))

    ## add exit function
    exit_func_partial = partial(utils.exit_func, logger, model_ema=model_ema, 
                                dataloader=dataloader, dataset=dataset)
    atexit.register(exit_func_partial)

    ## continue training
    if args.from_checkpoint:
        checkpoint = torch.load(args.ckpt_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model_ema.module.load_state_dict(checkpoint['model_state_dict_ema'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        _batch_size = checkpoint["batch_size"]
        crt_epoch = checkpoint["crt_epoch"]

        if batch_size != _batch_size:
            logger.warning(
                "!!!Warning!!! batch_size from checkpoint not match : {} != {}".format(batch_size, _batch_size)
            )
        logger.info("Checkpoint load from {}".format(args.ckpt_path))
        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        crt_epoch = 0

    ## training the model
    logger.info("Start training:")
    logger.info("Config path: {}".format(args.cfg_path))
    logger.info("Weights will be saved in experiment_dir = {}".format(args.exp_dir))

    training_epoch = config['training_config']['training_epoch']
    clip_grad_l2norm = config['training_config']['clip_grad_l2norm']
    log_interval = config['training_config']['log_interval']
    save_interval = config['training_config']['save_interval']
    eval_start_epoch = config['training_config']['eval_start_epoch']
    
    total_steps = 0
    model.train()
    for epoch in range(training_epoch):
        ## if resume, update total steps
        if epoch < crt_epoch:
            total_steps += len(dataloader)
            continue

        dataloader.sampler.set_epoch(epoch)

        logger.info("[Train]: Epoch {:d} started".format(epoch))
        losses_tracker = {}
        epoch_start = time.time()

        ## Training the model for one epoch
        for epoch_step, input_data in enumerate(dataloader):
            ## data to cuda
            input_data = utils.dict_to_device(input_data, device)

            training_lr = scheduler.get_last_lr()[-1]

            ## model forward
            loss_dict = model(input_data)
            
            ## backward
            optimizer.zero_grad(set_to_none=True)
            loss_dict['total_loss'].backward()
            if clip_grad_l2norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
            
            optimizer.step()
            scheduler.step()

            ## ema update
            model_ema.update(model)

            ## track all losses
            loss_dict = utils.reduce_loss(loss_dict)  # only for log
            for key, value in loss_dict.items():
                if key not in losses_tracker:
                    losses_tracker[key] = utils.AverageMeter()
                losses_tracker[key].update(value.item())

            if total_steps % log_interval == 0:
                # print to terminal
                block1 = "[Train]: [{:03d}][{:05d}/{:05d}]".format(epoch, epoch_step, len(dataloader) - 1)
                block2 = "Total loss={:.4f}".format(losses_tracker["total_loss"].avg)
                block3 = ["{:s}={:.4f}".format(key, value.avg) for key, value in losses_tracker.items() if key != "total_loss"]
                block4 = "training lr={:.1e}".format(training_lr)
                block5 = "mem={:.0f}MB".format(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
                logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5]))

            ## update steps
            total_steps += 1

        ## log epoch mean loss
        if args.rank == 0:
            logger.info("Epoch time: {:.4f}s".format(time.time() - epoch_start))

        ## save ckpt
        if (epoch + 1) % save_interval == 0 and (epoch + 1) >= eval_start_epoch:
            save_path = os.path.join(args.exp_dir, 'model_epoch_{}_{}.pth'.format(epoch + 1, args.data_name))
            if args.rank == 0:
                save_checkpoint(batch_size, epoch, model, optimizer, scheduler, save_path, model_ema=model_ema)
            logger.info("Checkpoint is saved: {}".format(save_path))

    ## save the last ckpt
    save_path = os.path.join(args.exp_dir, 'model_last.pth')
    if args.rank == 0:
        save_checkpoint(batch_size, epoch, model, optimizer, scheduler, save_path, model_ema=model_ema)
    logger.info("Checkpoint is saved: {}".format(save_path))
    logger.info(f"Log saved at {os.path.join(args.exp_dir, 'logfile')}")
    logger.info("Training Over...\n")

if __name__ == "__main__":
    main()