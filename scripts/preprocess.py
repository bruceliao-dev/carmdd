import argparse, os, json
import numpy as np
from tqdm import tqdm
from carmdd.data.dtw_align import resample, dtw_align

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["lmvd", "avec2019"])
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids = sorted(os.listdir(args.root))
    for sid in tqdm(ids):
        src = os.path.join(args.root, sid)
        if not os.path.isdir(src):
            continue
        vision = np.load(os.path.join(src, "vision.npy"))
        audio = np.load(os.path.join(src, "audio.npy"))
        text = np.load(os.path.join(src, "text.npy")) if os.path.exists(os.path.join(src, "text.npy")) else np.zeros((len(vision), 8), dtype=int)
        # resample to common frame rate
        T = len(vision)
        vision = resample(vision, T)
        audio = resample(audio, T)
        text = resample(text, T)
        v, a, t = dtw_align(vision, audio, text)
        dst = os.path.join(args.out, sid)
        os.makedirs(dst, exist_ok=True)
        np.save(os.path.join(dst, "vision.npy"), v.astype(np.float32))
        np.save(os.path.join(dst, "audio.npy"), a.astype(np.float32))
        np.save(os.path.join(dst, "text.npy"), t.astype(np.int64))
        if os.path.exists(os.path.join(src, "meta.json")):
            with open(os.path.join(src, "meta.json")) as f:
                meta = json.load(f)
            with open(os.path.join(dst, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()