import os, random, json, logging
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
    return logger

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

def compute_regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        pcc = float(pearsonr(y_true, y_pred)[0])
    except Exception:
        pcc = 0.0
    return {"rmse": rmse, "pcc": pcc}