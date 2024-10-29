import os

import torch
import torch.distributed as dist

from mega_core.utils import gpu_indices, ompi_size, ompi_rank, get_master_ip


def init_dist(launcher, args, backend='nccl'):
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, args)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, args)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def _init_dist_pytorch(backend, args):
    # os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    print('set group')
    torch.distributed.init_process_group(
        backend=backend, init_method="env://"
    )
    print('group set done.')

    # if args.distributed:
    #     args.rank = int(os.environ["RANK"])
    #     args.world_size = int(os.environ['WORLD_SIZE'])
    #     args.gpu = int(os.environ['LOCAL_RANK'])

    # torch.cuda.set_device(args.local_rank)
    # print('set group')
    # torch.distributed.init_process_group(backend=backend, init_method=args.dist_url,
    #                                      world_size=args.world_size, rank=args.rank)
    
    # # torch.distributed.init_process_group(backend=backend, init_method=args.dist_url,
    # #                                      world_size=args.world_size, rank=args.rank)
    # print('group set done.')
    
    # torch.distributed.barrier()
    # setup_for_distributed(args.rank == 0)

def _init_dist_mpi(backend, args):
    gpus = list(gpu_indices())
    gpu_num = len(gpus)
    world_size = ompi_size()
    rank = ompi_rank()
    dist_url = 'tcp://' + get_master_ip() + ':23456'
    torch.cuda.set_device(int(gpus[0]))  # Set current GPU to the first
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        group_name='mtorch')
    print(
        "World Size is {}, Backend is {}, Init Method is {}, rank is {}, gpu num is{}"
        .format(world_size, backend, dist_url, ompi_rank(), gpu_num))
