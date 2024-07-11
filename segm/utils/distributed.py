from pathlib import Path
import torch
import torch.distributed as dist

import segm.utils.torch as ptu


def init_process(backend="nccl"):
    print(f"Starting process with rank {ptu.dist_rank}...", flush=True)

    dist.init_process_group(
        backend=backend,
        rank=ptu.dist_rank,
        world_size=ptu.world_size,
    )
    print(f"Process {ptu.dist_rank} is connected.", flush=True)
    dist.barrier()

    silence_print(ptu.dist_rank == 0)
    if ptu.dist_rank == 0:
        print(f"All processes are connected.", flush=True)


def silence_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def sync_model(sync_dir, model):
    # https://github.com/ylabbe/cosypose/blob/master/cosypose/utils/distributed.py
    sync_path = Path(sync_dir).resolve() / "sync_model.pkl"
    if ptu.dist_rank == 0 and ptu.world_size > 1:
        torch.save(model.state_dict(), sync_path)
    dist.barrier()
    if ptu.dist_rank > 0:
        model.load_state_dict(torch.load(sync_path))
    dist.barrier()
    if ptu.dist_rank == 0 and ptu.world_size > 1:
        sync_path.unlink()
    return model
