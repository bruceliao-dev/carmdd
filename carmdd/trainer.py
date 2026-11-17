import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .losses import MultiTaskLoss
from .utils import set_seed, get_logger, save_json, compute_classification_metrics, compute_regression_metrics

class Trainer:
    def __init__(self, model, cfg, train_loader, val_loader, device="cuda"):
        self.model = model.to(device)
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.opt = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        self.loss_fn = MultiTaskLoss(cfg["alpha"], cfg["beta"])
        self.writer = SummaryWriter(cfg["output_dir"])
        self.logger = get_logger("CARMDD")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg["epochs"])

    def _step(self, batch, train=True):
        vision = batch["vision"].to(self.device)
        audio = batch["audio"].to(self.device)
        text = batch["text"]
        if text is not None:
            text = (text[0].to(self.device), text[1].to(self.device))
        labels = batch["label"].to(self.device)
        phq8 = batch["phq8"].to(self.device)

        with torch.set_grad_enabled(train):
            out = self.model(vision, audio, text)
            loss = self.loss_fn(out["logits"], out["severity"], labels, phq8)
            if train:
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
        probs = torch.sigmoid(out["logits"]).detach().cpu().numpy()
        return loss.item(), probs, labels.cpu().numpy(), phq8.cpu().numpy(), out["severity"]

    def train(self):
        best_val = float("inf")
        patience = 0
        for epoch in range(self.cfg["epochs"]):
            self.model.train()
            losses = []
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                loss, *_ = self._step(batch, train=True)
                losses.append(loss)
            self.scheduler.step()
            val_loss = self._validate()
            self.writer.add_scalar("train/loss", sum(losses) / len(losses), epoch)
            self.writer.add_scalar("val/loss", val_loss, epoch)
            self.logger.info(f"Epoch {epoch}: train={sum(losses)/len(losses):.4f} val={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                torch.save(self.model.state_dict(), os.path.join(self.cfg["output_dir"], "best.pt"))
            else:
                patience += 1
                if patience >= self.cfg["patience"]:
                    self.logger.info("Early stopping.")
                    break

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        losses = []
        for batch in self.val_loader:
            loss, *_ = self._step(batch, train=False)
            losses.append(loss)
        return sum(losses) / max(len(losses), 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_probs, all_labels, all_phq8, all_sev = [], [], [], []
        for batch in loader:
            loss, probs, labels, phq8, sev = self._step(batch, train=False)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
            all_phq8.extend(phq8.tolist())
            if sev is not None:
                all_sev.extend(sev.detach().cpu().numpy().tolist())
        import numpy as np
        metrics = compute_classification_metrics(np.array(all_labels), np.array(all_probs))
        if all_sev:
            metrics.update(compute_regression_metrics(np.array(all_phq8), np.array(all_sev)))
        return metrics