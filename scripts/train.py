import argparse, yaml, torch
from torch.utils.data import DataLoader
from carmdd.models.carmdd import CARMDD
from carmdd.data import MultimodalDepressionDataset, collate_fn
from carmdd.trainer import Trainer
from carmdd.utils import set_seed, get_logger

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    t_cfg = cfg["train"]
    m_cfg.update({
        "use_text": ds_cfg["use_text"],
        "use_regression": ds_cfg["use_regression"],
    })
    logger = get_logger("train")

    for seed in t_cfg["seeds"]:
        set_seed(seed)
        out_dir = f"{t_cfg['output_dir']}/seed{seed}"
        import os; os.makedirs(out_dir, exist_ok=True)
        train_ds = MultimodalDepressionDataset(ds_cfg["root"], "train", ds_cfg["use_text"], ds_cfg.get("phq8_threshold", 10))
        val_ds   = MultimodalDepressionDataset(ds_cfg["root"], "val",   ds_cfg["use_text"], ds_cfg.get("phq8_threshold", 10))
        train_loader = DataLoader(train_ds, batch_size=t_cfg["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4)
        val_loader   = DataLoader(val_ds,   batch_size=t_cfg["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=4)

        model = CARMDD(m_cfg)
        t_cfg_run = dict(t_cfg); t_cfg_run["output_dir"] = out_dir
        trainer = Trainer(model, t_cfg_run, train_loader, val_loader, device=args.device)
        trainer.train()
        metrics = trainer.evaluate(val_loader)
        logger.info(f"seed={seed} val metrics={metrics}")

if __name__ == "__main__":
    main()