import argparse, yaml, torch
from torch.utils.data import DataLoader
from carmdd.models.carmdd import CARMDD
from carmdd.data import MultimodalDepressionDataset, collate_fn
from carmdd.trainer import Trainer
from carmdd.utils import get_logger

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--transfer", default=None, choices=[None, "lmvd2avec"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    m_cfg.update({"use_text": ds_cfg["use_text"], "use_regression": ds_cfg["use_regression"]})

    # Transfer mode: source LMVD model evaluated on AVEC; disable text path
    if args.transfer == "lmvd2avec":
        m_cfg["use_text"] = False
        m_cfg["meter"] = False
        m_cfg["use_regression"] = False

    model = CARMDD(m_cfg).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    test_ds = MultimodalDepressionDataset(ds_cfg["root"], args.split, m_cfg["use_text"], ds_cfg.get("phq8_threshold", 10))
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=4)
    trainer = Trainer(model, cfg["train"], loader, loader, device=args.device)
    metrics = trainer.evaluate(loader)
    print(metrics)

if __name__ == "__main__":
    main()