import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

class MultimodalDepressionDataset(Dataset):
    """
    Expects per-sample folder:
        <root>/<id>/vision.npy  [T, D_v]
        <root>/<id>/audio.npy   [T, D_a]
        <root>/<id>/text.npy    [T, L]  (token ids, optional)
        <root>/<id>/meta.json   {"label": 0/1, "phq8": float, "split": "train|val|test"}
    """
    def __init__(self, root, split="train", use_text=True, phq8_threshold=10):
        self.root = root
        self.split = split
        self.use_text = use_text
        self.phq8_threshold = phq8_threshold
        self.ids = []
        for sid in sorted(os.listdir(root)):
            meta_path = os.path.join(root, sid, "meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("split") == split:
                self.ids.append(sid)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        d = os.path.join(self.root, sid)
        vision = torch.from_numpy(np.load(os.path.join(d, "vision.npy"))).float()
        audio = torch.from_numpy(np.load(os.path.join(d, "audio.npy"))).float()
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
        label = int(meta.get("label", int(meta.get("phq8", 0) >= self.phq8_threshold)))
        phq8 = float(meta.get("phq8", 0.0))
        text = None
        if self.use_text and os.path.exists(os.path.join(d, "text.npy")):
            text = torch.from_numpy(np.load(os.path.join(d, "text.npy"))).long()
        return {
            "vision": vision, "audio": audio, "text": text,
            "label": label, "phq8": phq8, "id": sid,
        }

def collate_fn(batch):
    max_T = max(b["vision"].shape[0] for b in batch)
    def pad(x):
        if x is None:
            return None
        T, *rest = x.shape
        if T >= max_T:
            return x[:max_T]
        pad_shape = (max_T - T, *rest)
        return torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype)], dim=0)
    vision = torch.stack([pad(b["vision"]) for b in batch])
    audio = torch.stack([pad(b["audio"]) for b in batch])
    has_text = all(b["text"] is not None for b in batch)
    if has_text:
        L = batch[0]["text"].shape[-1]
        text = torch.stack([pad(b["text"]) if b["text"].shape[1] == L else b["text"][:, :L] for b in batch])
        mask = (text != 0).any(-1)
    else:
        text, mask = None, None
    labels = torch.tensor([b["label"] for b in batch]).float()
    phq8 = torch.tensor([b["phq8"] for b in batch]).float()
    return {
        "vision": vision, "audio": audio,
        "text": (text, mask) if text is not None else None,
        "label": labels, "phq8": phq8,
        "ids": [b["id"] for b in batch],
    }