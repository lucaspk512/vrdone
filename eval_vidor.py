from tqdm import tqdm
import os
import argparse
import yaml
from collections import defaultdict
import atexit
from functools import partial
import json

import torch
from torch.utils import data

import utils.misc as utils
from utils.logging import setup_logger
from utils.evaluate import EvaluationFormatConvertor, eval_relation
from dataloaders.vidor import VidOR
from models.maskvrd import MaskVRD

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description="Test a Video Relation Detector")

    # control
    parser.add_argument("--cfg_path", type=str, help="...")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--ckpt_path", type=str, help="ckpt path to evaluate")
    parser.add_argument("--eval_exp_dir", default=False, action="store_true", help="the dir saving all ckpts")
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--scale", default=None, type=int)
    parser.add_argument("--eval_start_epoch", type=int, default=3)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval_file_name", type=str, default="eval")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--save_result", default=False, action="store_true")

    args = parser.parse_args()
    return args

def evaluate():  
    args = parse_args()

    ## load configs
    with open(args.cfg_path, 'r', encoding='utf-8') as f:
        cfg_file = f.read()
    config = yaml.safe_load(cfg_file)

    ## update config
    config['training_config']['training_epoch'] = args.epochs
    config['training_config']['eval_start_epoch'] = args.eval_start_epoch
    config['inference_config']['topk'] = args.topk
    config['dataset_config'].update(config['test_dataset_config'])

    ## set seed
    utils.set_seed(args.seed)
    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'logfile'), exist_ok=True)

    ## setup logger
    logger = setup_logger("Test", save_dir=os.path.join(args.exp_dir, 'logfile'), filename=args.eval_file_name + '_log.json')
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{json.dumps(config, indent=4)}")

    ## construct data
    dataset = VidOR(config['dataset_config'], args.scale)
    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        drop_last=False,
        collate_fn=dataset.collator_func,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    ## construct model
    device = f"cuda:0"
    model = MaskVRD(config['model_config'], device=device)
    model.eval()
    model._config_eval(config["inference_config"])
    model = model.to(device)

    ## add all of the ckpts
    training_epoch = config['training_config']['training_epoch']
    eval_start_epoch = config['training_config']['eval_start_epoch']
    eval_interval = config['training_config']['save_interval']
    ckpt_paths = []
    if args.eval_exp_dir:
        for epoch in range(eval_start_epoch - 1, training_epoch, eval_interval):
            cp = os.path.join(args.exp_dir, "model_epoch_{}_vidor.pth".format(epoch + 1))
            ckpt_paths.append(cp)
    else:
        assert args.ckpt_path
        ckpt_paths.append(args.ckpt_path)
    
    convertor = EvaluationFormatConvertor("vidor")
    all_results = defaultdict(list)

    ## metric keys
    metric_keys = [
        "RelDet_mAP", "RelDet_AR@50", "RelDet_AR@100", 
        "RelTag_AP@1", "RelTag_AP@5", "RelTag_AP@10"
    ]
    exit_func_partial = partial(utils.exit_func, logger, model=model, dataloader=dataloader, dataset=dataset)
    atexit.register(exit_func_partial)

    logger.info("Start inference...")
    for ckpt_idx, ckpt_path in enumerate(ckpt_paths):
        ## load ckpt
        logger.info("Loading checkpoint from: {}".format(ckpt_path))
        state_dict = torch.load(ckpt_path, map_location=device)
        
        if 'model_state_dict_ema' in state_dict:
            model_state_dict = state_dict['model_state_dict_ema'].copy()
        else:
            model_state_dict = state_dict['model_state_dict'].copy()

        rst_state_dict = {}
        start_with_module = list(model_state_dict.keys())[0].startswith('module.')
        for k, v in model_state_dict.items():
            if start_with_module:
                assert k.startswith('module.')
                rst_state_dict[k[7:]] = v
            else:
                assert not k.startswith('module.')
                rst_state_dict[k] = v

        model.load_state_dict(rst_state_dict)
        del state_dict, model_state_dict, rst_state_dict
        logger.info("Load done.")

        ## start eval
        predict_relations = dict()
        for proposal in tqdm(dataloader):
            if proposal is None:
                continue

            proposal = utils.dict_to_device(proposal, device)
            with torch.no_grad():
                batch_triplets = model(proposal)
            
            if batch_triplets is None:
                continue

            pr_result = convertor.to_eval_format_pr(proposal['video_name'], batch_triplets)
            predict_relations.update(pr_result)

        ## if no valid prediction
        if len(predict_relations.keys()) < 1:
            logger.info("None of valid prediction.")
            results = dict()
            for k in metric_keys:
                results[k] = 0.0

        else:
            # continue here
            results = eval_relation(
                dataset_type="vidor", 
                prediction_results=predict_relations, 
                config=config
            )

        ## print the metric
        for k, v in results.items():
            assert k in metric_keys
            all_results[k].append(v)
            logger.info("{}: {:.6f}".format(k, v))

        if args.save_result:
            save_path = os.path.join(args.exp_dir, 'VidORval_predict_relations_topk{}_epoch{}.json'.format(args.topk, ckpt_idx + eval_start_epoch))
            logger.info("Saving predict_relations into {}...".format(save_path))
            with open(save_path, 'w') as f:
                json.dump(predict_relations, f)
            logger.info("Predicted relations have been saved at {}".format(save_path))

    if len(ckpt_paths) > 1:
        logger.info("------------------------------------------------------------------------------------------")
        ## find the best rst
        for _key in metric_keys:
            max_index = all_results[_key].index(max(all_results[_key]))
            logger.info('Best {} result is in epoch {}'.format(_key, max_index * eval_interval + eval_start_epoch))
            for k, v in all_results.items():
                logger.info(k + ": {:.6f}".format(v[max_index]))
        
        logger.info('All of the results:')
        logger.info(all_results)
        
    logger.info("Log file have been saved at {}".format(os.path.join(args.exp_dir, 'logfile', args.eval_file_name + '_log.json')))
    logger.info('Eval done.')            

if __name__ == "__main__":
    evaluate()