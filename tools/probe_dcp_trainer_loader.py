import sys
from pathlib import Path

import yaml
from torchmetrics.functional import f1_score
import pytorch_lightning as pl

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "Deep_Correlated_Prompting-main"
sys.path.insert(0, str(CODE))

from clip.datamodules.multitask_datamodule import MTDataModule


class Dummy(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        return None


def main():
    hp = CODE / "pilot_logs" / "dcp_mmimdb_epoch1_pilot_002_seed0" / "version_0" / "hparams.yaml"
    config = yaml.safe_load(hp.read_text(encoding="utf-8"))["config"]
    dm = MTDataModule(config, dist=False)
    dm.setup("fit")
    print("before_trainer", len(dm.train_dataset), dm.batch_size, len(dm.train_dataloader()))
    trainer = pl.Trainer(
        gpus=1,
        num_nodes=1,
        precision=16,
        max_steps=0,
        logger=False,
        checkpoint_callback=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=128,
    )
    trainer.datamodule = dm
    loader = dm.train_dataloader()
    print("after_trainer_direct", len(dm.train_dataset), dm.batch_size, len(loader), loader.batch_size)


if __name__ == "__main__":
    main()
