import sys
sys.path.append("./segmenter")
from pathlib import Path
import yaml
import json
import torch
import click
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from segm.utils import distributed as distrib
import segm.utils.torch as ptu
from segm import config

from segm.model.factory import create_segmenter
from segm.optim.factory import create_optimizer, create_scheduler
from segm.data.factory import create_dataset

from timm.utils import NativeScaler
from contextlib import suppress

from segm.utils.distributed import sync_model
from segm.engine import train_one_epoch, evaluate


def translate_mae_to_segmenter(model_checkpoint, target_model):
    # Adapt key naming for preencoder model
    is_preencoder = False
    for key in list(model_checkpoint.keys()):
        if key.startswith("model."):
            model_checkpoint[key[6:]] = model_checkpoint.pop(key)
            is_preencoder = True
    if is_preencoder:
        model_checkpoint.pop("encoder.weight")
        model_checkpoint.pop("encoder.bias")

    # Delete surplus keys
    for key in list(model_checkpoint.keys()):
        if key.startswith("cls_token") or key.startswith("pos_embed") or key.startswith("patch_embed.") or key.startswith("blocks."):
            model_checkpoint["encoder."+key] = model_checkpoint.pop(key)
        elif key.startswith("decoder_") or key in ["mask_token", "norm.weight", "norm.bias"]:
            model_checkpoint.pop(key)

    # Add missing keys
    names = ["norm", "head"]
    for key in names:
        model_checkpoint[f"encoder.{key}.weight"] = getattr(target_model.encoder, key).weight.detach()
        model_checkpoint[f"encoder.{key}.bias"] = getattr(target_model.encoder, key).bias.detach()
    fnames = ["0.norm1", "0.norm2", "0.attn.qkv", "0.attn.proj", "0.mlp.fc1", "0.mlp.fc2"]
    names = fnames + [k.replace("0.", "1.") for k in fnames]
    for key in names:
        idx = int(key.split(".")[0])
        rest = key.split(".")[1:]
        cur = target_model.decoder.blocks[idx]
        for subkey in rest:
            cur = getattr(cur, subkey)
        model_checkpoint[f"decoder.blocks.{key}.weight"] = cur.weight.detach()
        model_checkpoint[f"decoder.blocks.{key}.bias"] = cur.bias.detach()
    for key in ["proj_dec", "decoder_norm", "mask_norm"]:
        model_checkpoint[f"decoder.{key}.weight"] = getattr(target_model.decoder, key).weight.detach()
        model_checkpoint[f"decoder.{key}.bias"] = getattr(target_model.decoder, key).bias.detach()
    model_checkpoint["decoder.cls_emb"] = target_model.decoder.cls_emb.detach()
    model_checkpoint["decoder.proj_patch"] = target_model.decoder.proj_patch.detach()
    model_checkpoint["decoder.proj_classes"] = target_model.decoder.proj_classes.detach()


@click.command(help="")
@click.option("--log-dir", type=str, help="logging directory")
@click.option("--dataset", type=str)
@click.option("--im-size", default=None, type=int, help="dataset resize size")
@click.option("--crop-size", default=None, type=int)
@click.option("--window-size", default=None, type=int)
@click.option("--window-stride", default=None, type=int)
@click.option("--backbone", default="", type=str)
@click.option("--decoder", default="", type=str)
@click.option("--optimizer", default="sgd", type=str)
@click.option("--scheduler", default="polynomial", type=str)
@click.option("--weight-decay", default=0.0, type=float)
@click.option("--dropout", default=0.0, type=float)
@click.option("--drop-path", default=0.1, type=float)
@click.option("--batch-size", default=None, type=int)
@click.option("--epochs", default=None, type=int)
@click.option("-lr", "--learning-rate", default=None, type=float)
@click.option("--normalization", default=None, type=str)
@click.option("--eval-freq", default=None, type=int)
@click.option("--amp/--no-amp", default=False, is_flag=True)
@click.option("--resume/--no-resume", default=True, is_flag=True)
@click.option("--mae", default=False, is_flag=True)
@click.option("--mae_chp", default="", type=str)
@click.option("--channels", default=3, type=int)
@click.option("--prefix", default="sth", type=str)
def main(
    log_dir,
    dataset,
    im_size,
    crop_size,
    window_size,
    window_stride,
    backbone,
    decoder,
    optimizer,
    scheduler,
    weight_decay,
    dropout,
    drop_path,
    batch_size,
    epochs,
    learning_rate,
    normalization,
    eval_freq,
    amp,
    resume,
    mae,
    mae_chp,
    channels,
    prefix,
):
    # start distributed mode
    ptu.set_gpu_mode(True)
    if ptu.distributed:
        distrib.init_process()
    torch.backends.cudnn.benchmark = True

    # set up configuration
    cfg = config.load_config()
    model_cfg = cfg["model"][backbone]
    dataset_cfg = cfg["dataset"][dataset]
    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]
    else:
        decoder_cfg = cfg["decoder"][decoder]

    # model config
    if not im_size:
        im_size = dataset_cfg["im_size"]
    if not crop_size:
        crop_size = dataset_cfg.get("crop_size", im_size)
    if not window_size:
        window_size = dataset_cfg.get("window_size", im_size)
    if not window_stride:
        window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    # dataset config
    batch_size = dataset_cfg["batch_size"]
    num_epochs = dataset_cfg["epochs"]
    lr = dataset_cfg["learning_rate"]
    if batch_size:
        world_batch_size = batch_size
    if epochs:
        num_epochs = epochs
    if learning_rate:
        lr = learning_rate
    if eval_freq is None:
        eval_freq = dataset_cfg.get("eval_freq", 1)

    if normalization:
        model_cfg["normalization"] = normalization

    # experiment config
    world_batch_size = batch_size * ptu.world_size
    variant = dict(
        world_batch_size=world_batch_size,
        version="normal",
        resume=resume,
        dataset_kwargs=dict(
            dataset=dataset,
            image_size=im_size,
            crop_size=crop_size,
            batch_size=batch_size,
            normalization=model_cfg["normalization"],
            split="train",
            num_workers=10,
        ),
        algorithm_kwargs=dict(
            batch_size=batch_size,
            start_epoch=0,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
        ),
        optimizer_kwargs=dict(
            opt=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            clip_grad=None,
            sched=scheduler,
            epochs=num_epochs,
            min_lr=1e-5,
            poly_power=0.9,
            poly_step_size=1,
        ),
        net_kwargs=model_cfg,
        amp=amp,
        log_dir=log_dir,
        inference_kwargs=dict(
            im_size=im_size,
            window_size=window_size,
            window_stride=window_stride,
        ),
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if mae:
        start_checkpoint_path = log_dir / "mae_pretrain_vit_large.pth"
        if mae_chp:
            start_checkpoint_path = Path(mae_chp)
    save_checkpoint_path = log_dir

    # dataset
    dataset_kwargs = variant["dataset_kwargs"]

    train_loader = create_dataset(dataset_kwargs)
    val_kwargs = dataset_kwargs.copy()
    val_kwargs["split"] = "val"
    val_kwargs["batch_size"] = 1
    val_kwargs["crop"] = False
    val_loader = create_dataset(val_kwargs)
    n_cls = train_loader.unwrapped.n_cls

    # model
    net_kwargs = variant["net_kwargs"]
    net_kwargs["n_cls"] = n_cls
    net_kwargs["channels"] = channels
    
    model = create_segmenter(net_kwargs)
    model.to(ptu.device)

    # optimizer
    optimizer_kwargs = variant["optimizer_kwargs"]
    optimizer_kwargs["iter_max"] = len(train_loader) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v
    optimizer = create_optimizer(opt_args, model)
    lr_scheduler = create_scheduler(opt_args, optimizer)
    amp_autocast = suppress
    loss_scaler = None
    if amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    # resume
    if resume and start_checkpoint_path.exists():
        print(f"Resuming training from checkpoint: {start_checkpoint_path}")
        checkpoint = torch.load(start_checkpoint_path, map_location="cpu")
        if mae:
            translate_mae_to_segmenter(checkpoint["model"], model)
            variant["algorithm_kwargs"]["start_epoch"] = 0
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            variant["algorithm_kwargs"]["start_epoch"] = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"])
        if loss_scaler and "loss_scaler" in checkpoint:
            loss_scaler.load_state_dict(checkpoint["loss_scaler"])
    else:
        pass
        # raise Exception("This shouldn't happen because we only resume from pretraining checkpoints")
        # sync_model(log_dir, model)

    if ptu.distributed:
        model = DDP(model, device_ids=[ptu.device], find_unused_parameters=True)

    # save config
    variant["net_kwargs"] = net_kwargs
    variant["dataset_kwargs"] = dataset_kwargs
    log_dir.mkdir(parents=True, exist_ok=True)

    # train
    start_epoch = variant["algorithm_kwargs"]["start_epoch"]
    num_epochs = variant["algorithm_kwargs"]["num_epochs"]
    eval_freq = variant["algorithm_kwargs"]["eval_freq"]

    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    val_seg_gt = val_loader.dataset.get_gt_seg_maps()

    epoch = 0
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch
        train_logger = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            epoch,
            amp_autocast,
            loss_scaler,
        )

        # evaluate
        eval_epoch = epoch % eval_freq == 0 or epoch == num_epochs - 1
        if eval_epoch:
            eval_logger = evaluate(
                model,
                val_loader,
                val_seg_gt,
                window_size,
                window_stride,
                amp_autocast,
                prefix,
            )
            print(f"Stats [{epoch}]:", eval_logger, flush=True)
            print("")

        # log stats
        if ptu.dist_rank == 0:
            train_stats = {
                k: meter.global_avg for k, meter in train_logger.meters.items()
            }
            val_stats = {}
            if eval_epoch:
                val_stats = {
                    k: meter.global_avg for k, meter in eval_logger.meters.items()
                }

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
                "epoch": epoch,
                "num_updates": (epoch + 1) * len(train_loader),
            }

            with open(log_dir / f"log_{start_checkpoint_path.name}.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # save checkpoint
    if ptu.dist_rank == 0:
        snapshot = dict(
            model=model_without_ddp.state_dict(),
            optimizer=optimizer.state_dict(),
            n_cls=model_without_ddp.n_cls,
            lr_scheduler=lr_scheduler.state_dict(),
        )
        if loss_scaler is not None:
            snapshot["loss_scaler"] = loss_scaler.state_dict()
        snapshot["epoch"] = epoch
        torch.save(snapshot, save_checkpoint_path/start_checkpoint_path.name)

    if ptu.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
