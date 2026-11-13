"""Shared settings: paths, character set and model/training defaults."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_ZIP = ROOT / "data" / "dataset.zip"
DATASET_DIR = ROOT / "dataset"            # extracted here, git-ignored
MODEL_PATH = ROOT / "models" / "ocr_crnn.pt"
STATE_PATH = ROOT / "models" / "train_state.pt"   # resume file, git-ignored
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
