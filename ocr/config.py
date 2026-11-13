"""Shared settings: paths, character set and model/training defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_ZIP = ROOT / "data" / "dataset.zip"
DATASET_DIR = ROOT / "dataset"            # extracted here, git-ignored
# The CNN is shared; the sequence model after it is one of these.
ARCHS = ("lstm", "transformer", "mamba")
ARCH_NAMES = {"lstm": "CNN + BiLSTM", "transformer": "CNN + Transformer", "mamba": "CNN + Mamba"}
MODEL_PATHS = {"lstm": ROOT / "models" / "ocr_crnn.pt",          # the original trained model
               "transformer": ROOT / "models" / "ocr_transformer.pt",
               "mamba": ROOT / "models" / "ocr_mamba.pt"}
STATE_PATHS = {a: ROOT / "models" / f"train_state_{a}.pt" for a in ARCHS}   # resume files, git-ignored
SAMPLES_DIR = ROOT / "samples"
WEB_DIR = ROOT / "ocr" / "web"

# Index 0 of the model output is reserved for the CTC "blank" symbol.
CHARSET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'\"()-/&@#%$+=*"
)

IMG_HEIGHT = 40        # every text line is resized to this height
MAX_WIDTH = 768       # longer training lines are squeezed to this width

EPOCHS = 30
STEPS_PER_EPOCH = 250
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
