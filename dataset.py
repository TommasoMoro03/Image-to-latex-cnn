import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle  # Used for loading vocab.pkl
import re
from tqdm import tqdm

from build_vocab import Vocab

# --- Configuration (Adjust these paths relative to where your main script will run) ---
PROCESSED_DATA_DIR = "./processed_data"
PROCESSED_IMAGES_TRAIN_DIR = os.path.join(PROCESSED_DATA_DIR, 'train_images')
PROCESSED_IMAGES_VALID_DIR = os.path.join(PROCESSED_DATA_DIR, 'val_images')
PROCESSED_IMAGES_TEST_DIR = os.path.join(PROCESSED_DATA_DIR, 'test_images')

TRAIN_PROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, 'train_processed.csv')
VALID_PROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, 'val_processed.csv')
TEST_PROCESSED_CSV = os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv')

VOCAB_FILE = os.path.join(PROCESSED_DATA_DIR, 'vocab.pkl')
MAX_SEQ_LEN_FILE = os.path.join(PROCESSED_DATA_DIR, 'max_seq_len.txt')

# --- Special Token IDs (Must match build_vocab.py) ---
START_TOKEN_ID = 0  # <s>
PAD_TOKEN_ID = 1  # <pad>
END_TOKEN_ID = 2  # </s>
UNK_TOKEN_ID = 3  # <unk>

# --- LaTeX Tokenization Pattern (MUST BE CONSISTENT ACROSS ALL FILES) ---
latex_tokenizer_pattern = re.compile(r"(\\\w+|\{|\}|\[|\]|\(|\)|\+|\-|\*|=|#|\$|&|%|~|_|\^|<|>|\||\!|`|,|;|\.|/|\\|\S)")


def _base_tokenize_latex(formula):
    """
    Core tokenization logic without adding special START/END tokens.
    This function should be identical across build_vocab.py and dataset.py
    for consistent token extraction.
    """
    if formula.startswith('"') and formula.endswith('"'):
        formula = formula[1:-1]

    formula = formula.strip()
    formula = re.sub(r'\s+', ' ', formula)

    tokens = [m.group(0) for m in latex_tokenizer_pattern.finditer(formula)]
    return tokens


def tokenize_latex_for_dataset(formula):
    """
    Tokenizes a LaTeX formula string and adds START/END tokens.
    Used specifically for preparing sequences for the model.
    """
    tokens = _base_tokenize_latex(formula)
    return ["<s>"] + tokens + ["</s>"]


# --- Helper function for formula to IDs ---
def formula_to_ids(formula_str, vocab_obj):
    """
    Converts a LaTeX formula string to a list of token IDs,
    including START and END tokens implicitly via tokenize_latex_for_dataset.
    """
    tokens = tokenize_latex_for_dataset(formula_str)
    return [vocab_obj.sign2id.get(token, UNK_TOKEN_ID) for token in tokens]


# --- Custom PyTorch Dataset Class ---
class Im2LatexDataset(Dataset):
    """
    PyTorch Dataset for loading image-LaTeX formula pairs.
    """

    def __init__(self, csv_filepath, images_dir, vocab_obj, max_seq_len, transform=None):
        self.images_dir = images_dir
        self.vocab = vocab_obj
        self.max_seq_len = max_seq_len
        self.transform = transform

        df = pd.read_csv(csv_filepath)
        self.data = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Loading {os.path.basename(csv_filepath)} data"):
            img_name = row['image_name']
            formula = str(row['formula'])
            img_path = os.path.join(self.images_dir, img_name)
            self.data.append((img_path, formula))

        print(f"Dataset initialized with {len(self.data)} samples from {csv_filepath}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, formula_str = self.data[idx]

        try:
            image = Image.open(img_path).convert('L')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}. Error: {e}. Returning dummy data.")
            dummy_image = torch.zeros(1, 160, 800)
            dummy_formula_ids = [START_TOKEN_ID, END_TOKEN_ID] + [PAD_TOKEN_ID] * (self.max_seq_len - 2)
            dummy_formula_tensor = torch.tensor(dummy_formula_ids, dtype=torch.long)
            return dummy_image, dummy_formula_tensor[:-1], dummy_formula_tensor[1:]

        formula_ids = formula_to_ids(formula_str, self.vocab)

        if len(formula_ids) < self.max_seq_len:
            padding_needed = self.max_seq_len - len(formula_ids)
            formula_ids.extend([PAD_TOKEN_ID] * padding_needed)
        else:
            formula_ids = formula_ids[:self.max_seq_len - 1] + [END_TOKEN_ID]

        formula_tensor = torch.tensor(formula_ids, dtype=torch.long)

        input_ids = formula_tensor[:-1]
        target_ids = formula_tensor[1:]

        return image, input_ids, target_ids


# --- Function to set up DataLoaders ---
def get_dataloaders(batch_size, num_workers=0, target_img_size=(800, 160)):
    """
    Initializes and returns train, validation, and test DataLoaders.

    Args:
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of subprocesses to use for data loading. 0 means main process only.
        target_img_size (tuple): (width, height) to resize images to.

    Returns:
        tuple: (train_loader, valid_loader, test_loader, vocab_obj, max_seq_len)
    """
    print(f"Attempting to load vocabulary from: {VOCAB_FILE}")
    # Directly use Vocab.load from the imported build_vocab module
    vocab_obj = Vocab.load(VOCAB_FILE)

    print(f"Attempting to load max_seq_len from: {MAX_SEQ_LEN_FILE}")
    with open(MAX_SEQ_LEN_FILE, 'r') as f:
        max_seq_len = int(f.read().strip())
    print(f"Max sequence length loaded: {max_seq_len}")

    image_transform = transforms.Compose([
        transforms.Resize(target_img_size[::-1]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    print("\nInitializing datasets...")
    train_dataset = Im2LatexDataset(
        csv_filepath=TRAIN_PROCESSED_CSV,
        images_dir=PROCESSED_IMAGES_TRAIN_DIR,
        vocab_obj=vocab_obj,
        max_seq_len=max_seq_len,
        transform=image_transform
    )

    valid_dataset = Im2LatexDataset(
        csv_filepath=VALID_PROCESSED_CSV,
        images_dir=PROCESSED_IMAGES_VALID_DIR,
        vocab_obj=vocab_obj,
        max_seq_len=max_seq_len,
        transform=image_transform
    )

    test_dataset = Im2LatexDataset(
        csv_filepath=TEST_PROCESSED_CSV,
        images_dir=PROCESSED_IMAGES_TEST_DIR,
        vocab_obj=vocab_obj,
        max_seq_len=max_seq_len,
        transform=image_transform
    )

    print("\nCreating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    print(f"DataLoaders created with batch size {batch_size} and {num_workers} workers.")
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}, Test batches: {len(test_loader)}")

    return train_loader, valid_loader, test_loader, vocab_obj, max_seq_len


# --- Example Usage (for testing this module directly) ---
def execute():
    print("Running dataset.py directly for testing...")

    TARGET_IMG_WIDTH_EXAMPLE = 800
    TARGET_IMG_HEIGHT_EXAMPLE = 160

    train_loader, valid_loader, test_loader, vocab_obj, max_seq_len = get_dataloaders(
        batch_size=16,
        num_workers=0,
        target_img_size=(TARGET_IMG_WIDTH_EXAMPLE, TARGET_IMG_HEIGHT_EXAMPLE)
    )

    print("\nTesting one batch from the training DataLoader:")
    for images, input_ids, target_ids in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Input IDs batch shape: {input_ids.shape}")
        print(f"Target IDs batch shape: {target_ids.shape}")

        print(f"\nFirst image (tensor) min/max: {images[0].min():.2f}/{images[0].max():.2f}")
        print(f"First formula input IDs: {input_ids[0]}")
        print(f"First formula target IDs: {target_ids[0]}")

        decoded_input_tokens = [
            vocab_obj.id2sign.get(idx.item(), '<unk>')
            for idx in input_ids[0]
            if idx.item() not in [PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID]
        ]
        decoded_target_tokens = [
            vocab_obj.id2sign.get(idx.item(), '<unk>')
            for idx in target_ids[0]
            if idx.item() not in [PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID]
        ]
        print(f"First formula input (decoded, clean): {''.join(decoded_input_tokens)}")
        print(f"First formula target (decoded, clean): {''.join(decoded_target_tokens)}")
        break


if __name__ == "__main__":
    execute()