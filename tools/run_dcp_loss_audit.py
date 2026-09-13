import math
import sys
from pathlib import Path

import torch
import yaml
from torchmetrics.functional import f1_score
import pytorch_lightning as pl

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "Deep_Correlated_Prompting-main"
sys.path.insert(0, str(CODE))

from clip.datamodules.multitask_datamodule import MTDataModule
from clip.modules import CLIPransformerSS


HPARAMS_PATH = CODE / "pilot_logs" / "dcp_mmimdb_epoch1_pilot_002_seed0" / "version_0" / "hparams.yaml"


class AuditCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        loader = trainer.train_dataloader
        print(
            "AUDIT_TRAIN_START",
            f"trainer_num_training_batches={trainer.num_training_batches}",
            f"trainer_loader_len={len(loader)}",
            f"trainer_loader_batch_size={loader.batch_size}",
            f"trainer_dataset_len={len(loader.dataset)}",
            flush=True,
        )


def main():
    config = yaml.safe_load(HPARAMS_PATH.read_text(encoding="utf-8"))["config"]
    config["exp_name"] = "dcp_mmimdb_loss_audit_001"
    config["max_steps"] = 1
    config["max_epoch"] = 1000
    config["fast_dev_run"] = False
    config["val_check_interval"] = 1.0
    config["num_workers"] = 0

    dm = MTDataModule(config, dist=False)
    original_dm_train_dataloader = MTDataModule.train_dataloader

    def traced_dm_train_dataloader(self):
        loader = original_dm_train_dataloader(self)
        print(
            "DM_LOADER_CALL",
            f"dataset_len={len(self.train_dataset)}",
            f"loader_len={len(loader)}",
            f"batch_size={loader.batch_size}",
            flush=True,
        )
        return loader

    MTDataModule.train_dataloader = traced_dm_train_dataloader
    model = CLIPransformerSS(config)
    original_training_step = model.training_step
    raw_losses = []

    def audited_training_step(batch, batch_idx):
        loss = original_training_step(batch, batch_idx)
        value = loss.detach().float()
        finite = bool(torch.isfinite(value).all())
        if batch_idx < 260:
            print(
                "RAW_LOSS",
                f"batch_idx={batch_idx}",
                f"value={value.item():.9f}",
                f"finite={finite}",
                f"dtype={loss.dtype}",
                flush=True,
            )
        raw_losses.append(value.item())
        return loss

    model.training_step = audited_training_step
    dm.setup("fit")
    print(
        "AUDIT_PREFLIGHT",
        f"dataset_len={len(dm.train_dataset)}",
        f"loader_len={len(dm.train_dataloader())}",
        f"loader_batch_size={dm.train_dataloader().batch_size}",
        flush=True,
    )
    trainer = pl.Trainer(
        gpus=config["num_gpus"],
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        accelerator=None,
        benchmark=True,
        deterministic=True,
        max_epochs=config["max_epoch"],
        max_steps=config["max_steps"],
        logger=False,
        checkpoint_callback=False,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=config["batch_size"] // config["per_gpu_batchsize"],
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        val_check_interval=config["val_check_interval"],
        weights_summary=None,
        callbacks=[AuditCallback()],
    )
    trainer.fit(model, datamodule=dm)

    finite_values = [x for x in raw_losses if math.isfinite(x)]
    print("AUDIT_SUMMARY")
    print(f"audit_training_batches={len(raw_losses)}")
    print(f"optimizer_steps={trainer.global_step}")
    print(f"raw_loss_min={min(finite_values) if finite_values else None}")
    print(f"raw_loss_max={max(finite_values) if finite_values else None}")
    print(f"raw_loss_mean={sum(finite_values) / len(finite_values) if finite_values else None}")
    print(f"raw_loss_nonfinite={len(raw_losses) - len(finite_values)}")
    print(f"cuda_available={torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
