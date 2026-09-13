import math
import sys
from pathlib import Path

import pyarrow as pa
import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code" / "Deep_Correlated_Prompting-main"
sys.path.insert(0, str(CODE))

from clip.datamodules.multitask_datamodule import MTDataModule


CONFIG_PATH = ROOT / "experiments" / "dcp_mmimdb_epoch1_pilot_002" / "print_config.txt"
HPARAMS_PATH = CODE / "pilot_logs" / "dcp_mmimdb_epoch1_pilot_002_seed0" / "version_0" / "hparams.yaml"


def arrow_rows(root, name):
    path = Path(root) / (name + ".arrow")
    reader = pa.ipc.RecordBatchFileReader(pa.memory_map(str(path), "r"))
    return reader.read_all().num_rows


def report_dataset(label, dataset, loader):
    datasets = getattr(dataset, "datasets", [dataset])
    counts = [len(texts) for child in datasets for texts in child.all_texts]
    print(f"{label}_dataset_length={len(dataset)}")
    print(f"{label}_dataloader_length={len(loader)}")
    print(f"{label}_plot_total={sum(counts)}")
    print(f"{label}_plot_average={sum(counts) / len(counts):.9f}")
    print(f"{label}_plot_min={min(counts)}")
    print(f"{label}_plot_max={max(counts)}")
    print(f"{label}_movies_one_plot={sum(c == 1 for c in counts)}")
    print(f"{label}_movies_multiple_plots={sum(c > 1 for c in counts)}")


def main():
    config = yaml.safe_load(HPARAMS_PATH.read_text(encoding="utf-8"))["config"]
    print(f"config_path={HPARAMS_PATH}")
    print(f"data_root={config['data_root']}")
    print(f"arrow_train_rows={arrow_rows(config['data_root'], 'mmimdb_train')}")
    print(f"arrow_dev_rows={arrow_rows(config['data_root'], 'mmimdb_dev')}")
    print(f"arrow_test_rows={arrow_rows(config['data_root'], 'mmimdb_test')}")

    dm = MTDataModule(config, dist=False)
    dm.setup("fit")
    print(f"per_gpu_batchsize={config['per_gpu_batchsize']}")
    print(f"accumulate_grad_batches={config['batch_size'] // config['per_gpu_batchsize']}")
    report_dataset("train", dm.train_dataset, dm.train_dataloader())
    report_dataset("val", dm.val_dataset, dm.val_dataloader())
    report_dataset("test", dm.test_dataset, dm.test_dataloader())
    train_batches = len(dm.train_dataloader())
    val_batches = len(dm.val_dataloader())
    print(f"theoretical_optimizer_steps={math.ceil(train_batches / 128)}")
    print(f"lightning_main_progress_total_train_plus_val={train_batches + val_batches}")


if __name__ == "__main__":
    main()
