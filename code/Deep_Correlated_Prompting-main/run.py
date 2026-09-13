import os
import copy
from torchmetrics.functional import f1_score
import pytorch_lightning as pl

from clip.config import ex
from clip.modules import CLIPransformerSS
from clip.modules.reliability_callbacks import GatePhaseModelCheckpoint
from clip.datamodules.multitask_datamodule import MTDataModule
import shutil
from torch.utils.data import DataLoader


def build_checkpoint_callback(config, checkpoint_dir):
    callback_kwargs = {
        "dirpath": checkpoint_dir,
        "save_top_k": 1,
        "verbose": True,
        "monitor": "val/the_metric",
        "mode": "max",
        "save_last": True,
    }
    if config.get("reliability_enabled", False):
        return GatePhaseModelCheckpoint(
            adapter_train_epochs=config.get("adapter_train_epochs", 10),
            **callback_kwargs
        )
    return pl.callbacks.ModelCheckpoint(**callback_kwargs)


def artifact_directories(logger, project_code_dir, run_name):
    version = str(logger.version)
    version_name = version if version.startswith("version_") else "version_{}".format(version)
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    snapshot_dir = os.path.join(
        project_code_dir, "result_model_files", run_name, version_name
    )
    return checkpoint_dir, snapshot_dir


def snapshot_source_tree(project_code_dir, target_dir):
    os.makedirs(target_dir, exist_ok=False)
    for filename in ("run.py", "requirements.txt"):
        source = os.path.join(project_code_dir, filename)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(target_dir, filename))
    for source_root_name in ("clip", "tools"):
        source_root = os.path.join(project_code_dir, source_root_name)
        for current_root, directories, filenames in os.walk(source_root):
            directories[:] = [
                name for name in directories if name != "__pycache__"
            ]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                source = os.path.join(current_root, filename)
                relative = os.path.relpath(source, project_code_dir)
                destination = os.path.join(target_dir, relative)
                os.makedirs(os.path.dirname(destination), exist_ok=True)
                shutil.copy2(source, destination)


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Keep all paths usable on Windows.  The original implementation treated
    # log_dir as both a logger path and a relative source-snapshot path, which
    # turned an absolute Windows path into './result_model_files/D:'.
    project_code_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = _config["log_dir"]
    if not os.path.isabs(log_dir):
        log_dir = os.path.abspath(os.path.join(project_code_dir, log_dir))
    _config["log_dir"] = log_dir

    resume_from = _config["resume_from"]
    if resume_from is not None and not os.path.isfile(resume_from):
        raise FileNotFoundError(
            "resume_from checkpoint does not exist: {}".format(resume_from)
        )

    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], int)
        else len(_config["num_gpus"])
    )
    use_ddp = num_gpus > 1 or _config["num_nodes"] > 1
    trainer_accelerator = "ddp" if use_ddp else None

    dm = MTDataModule(_config, dist=use_ddp)

    model = CLIPransformerSS(_config)
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(log_dir, exist_ok=True)
    run_name = f'{exp_name}_seed{_config["seed"]}'
    #logger = pl.loggers.TensorBoardLogger(
    #    _config["log_dir"],
    #    name=f'{exp_name}_seed{_config["seed"]}}',
    #)
    logger = pl.loggers.CSVLogger(
        log_dir,
        name=run_name,
    )
    os.makedirs(logger.log_dir, exist_ok=True)
    checkpoint_dir, target_dir = artifact_directories(
        logger, project_code_dir, run_name
    )
    checkpoint_callback = build_checkpoint_callback(_config, checkpoint_dir)
    snapshot_source_tree(project_code_dir, target_dir)

    callbacks = [checkpoint_callback]
#     from pytorch_lightning.profiler import SimpleProfiler
#     profiler = SimpleProfiler()
    
    grad_steps = _config["batch_size"] // (
        _config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]
    )
    print(_config["batch_size"], _config["per_gpu_batchsize"], num_gpus, _config["num_nodes"])
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None

    trainer = pl.Trainer(
        gpus=_config["num_gpus"],
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator=trainer_accelerator,
        benchmark=True,
        deterministic=True,
        max_epochs=_config["max_epoch"] if max_steps is None else 1000,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=resume_from,
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        limit_train_batches=_config.get("limit_train_batches", 1.0),
        limit_val_batches=_config.get("limit_val_batches", 1.0),
        val_check_interval=_config["val_check_interval"],
#         profiler=profiler,
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)
