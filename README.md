# CARMDD: Cross-modal Alignment and Refinement for Multimodal Depression Detection

Official implementation of the paper *"Cross-modal Alignment and Refinement for Multimodal Depression Detection"* (MTAP).

CARMDD is an end-to-end framework for automated depression screening from video interviews. The pipeline consists of four modules:

1. **Multimodal Alignment Encoding** — BGE-M3 (text) + ViT (vision) + HTS-AT (audio), with a METER-style vision–language fuser and DTW-based temporal alignment.
2. **Hierarchical Gated Fusion (HGF)** — bidirectional cross-modal gated attention at feature, segment, and global levels.
3. **Memory-Augmented Refinement** — external memory matrix with top-k read, Transformer-based reasoning, and erase/add write.
4. **Multi-task Prediction Head** — joint binary classification (BCE) and severity regression (smooth L1).

---

## Repository Structure
CARMDD/
├── README.md
├── requirements.txt
├── setup.py
├── configs/ # dataset-specific YAML configs
├── carmdd/
│ ├── models/ # encoders, METER fuser, HGF, memory, heads, full model
│ ├── data/ # dataset, DTW alignment
│ ├── losses.py # BCE + smooth L1
│ ├── trainer.py # training / evaluation loop
│ └── utils.py # logging, seeding, metrics
├── scripts/
│ ├── preprocess.py # feature extraction + DTW alignment
│ ├── train.py # training entrypoint
│ └── evaluate.py # evaluation + cross-dataset transfer
└── docker/Dockerfile


---

## Installation

### Option A — pip

```bash
git clone https://github.com/bruceliao/CARMDD.git
cd CARMDD
pip install -r requirements.txt
pip install -e .

### Option B — Docker
docker build -t carmdd:latest -f docker/Dockerfile .
docker run --gpus all -it --rm -v $PWD:/workspace carmdd:latest

requirements.txt pins PyTorch, transformers, fastdtw, scikit-learn, pyyaml, and tensorboard.

## Data Preparation
CARMDD expects pre-extracted features for each modality. The preprocessing script resamples all streams to 30 Hz and applies DTW between vision and audio (see Sec. 3.2.2 of the paper).

python scripts/preprocess.py \
    --dataset lmvd \
    --root /path/to/LMVD \
    --out  /path/to/LMVD_features \
    --fps 30

python scripts/preprocess.py \
    --dataset avec2019 \
    --root /path/to/AVEC2019 \
    --out  /path/to/AVEC2019_features \
    --fps 30

#Training
LMVD (binary classification only)
python scripts/train.py --config configs/lmvd.yaml

AVEC 2019 (multi-task: classification + PHQ-8 regression)
python scripts/train.py --config configs/avec2019.yaml


#Evaluation
bash
# In-domain evaluation
python scripts/evaluate.py --config configs/lmvd.yaml --ckpt runs/lmvd/best.pt

# AVEC 2019 with PHQ-8 regression
python scripts/evaluate.py --config configs/avec2019.yaml --ckpt runs/avec/best.pt

# Cross-dataset transfer: LMVD -> AVEC 2019 (Sec. 4.9)
python scripts/evaluate.py \
    --config configs/avec2019.yaml \
    --ckpt   runs/lmvd/best.pt \
    --transfer lmvd2avec

# Model Overview
video ──► ViT ──┐
                ├─► METER-Fuser ──┐
text  ──► BGE-M3┘                 │
                                  ├─► HGF (feature/segment/global) ──► Memory ──► Heads
audio ──► HTS-AT ─► W_a ─► DTW ──┘
