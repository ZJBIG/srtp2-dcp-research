"""Checkpoint policy for the final two-phase Reliability-DCP run."""

import pytorch_lightning as pl


class GatePhaseModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """Save last throughout training, but rank best only in the Gate phase."""

    def __init__(self, adapter_train_epochs, **kwargs):
        super().__init__(**kwargs)
        self.adapter_train_epochs = int(adapter_train_epochs)

    def save_checkpoint(self, trainer, pl_module):
        if trainer.current_epoch >= self.adapter_train_epochs:
            return super().save_checkpoint(trainer, pl_module)
        if trainer.fast_dev_run or trainer.running_sanity_check:
            return
        if self.last_global_step_saved == trainer.global_step:
            return
        self.last_global_step_saved = trainer.global_step
        candidates = self._monitor_candidates(trainer)
        self._save_last_checkpoint(trainer, pl_module, candidates)
