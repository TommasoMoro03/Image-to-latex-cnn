import os
import pandas as pd
from collections import Counter
import pickle
import re
from tqdm import tqdm

# --- Special Token IDs ---
START_TOKEN = 0  # <s>
PAD_TOKEN = 1  # <pad>
END_TOKEN = 2  # <eos>
UNK_TOKEN = 3  # <unk>

special_tokens_map = {
    "<s>": START_TOKEN,
    "<pad>": PAD_TOKEN,
    "</s>": END_TOKEN,
    "<unk>": UNK_TOKEN
}

# --- LaTeX Tokenization Pattern (MUST BE CONSISTENT ACROSS ALL FILES) ---
# This pattern extracts LaTeX commands (e.g., \frac), special characters,
# and individual characters/numbers.
latex_tokenizer_pattern = re.compile(r"(\\\w+|\{|\}|\[|\]|\(|\)|\+|\-|\*|=|#|\$|&|%|~|_|\^|<|>|\||\!|`|,|;|\.|/|\\|\S)")


def tokenize_latex_for_vocab(formula):
    """
    Tokenizes a LaTeX formula string using the robust regex pattern.
    This function is specifically for vocabulary building to ensure consistency.
    """
    # Remove potential outer quotes if present (common in CSV parsing)
    if formula.startswith('"') and formula.endswith('"'):
        formula = formula[1:-1]

    # Normalize internal whitespace before tokenizing to avoid empty tokens if regex splits on space
    formula = formula.strip()
    # The regex itself handles many whitespace cases, but this cleanup ensures consistency
    formula = re.sub(r'\s+', ' ', formula)

    # Use the pre-compiled regex pattern to find all tokens
    tokens = [m.group(0) for m in latex_tokenizer_pattern.finditer(formula)]

    # Do NOT add SOS/EOS here during vocabulary building, only when preparing sequence for model input.
    # The vocabulary should contain the raw tokens.
    return tokens


# --- Vocab Class Definition ---
class Vocab:
    def __init__(self, min_freq=1):
        self.sign2id = special_tokens_map.copy()
        self.id2sign = {idx: token for token, idx in self.sign2id.items()}
        self.length = len(self.sign2id)
        self.min_freq = min_freq

    def build(self, formulas):
        """
        Builds the vocabulary from a list of LaTeX formulas using the robust tokenization.
        """
        counter = Counter()
        for formula in tqdm(formulas, desc="Counting token frequencies"):
            # Use the consistent tokenization function
            tokens = tokenize_latex_for_vocab(formula)
            counter.update(tokens)

        # Add tokens to vocabulary based on minimum frequency
        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.sign2id:
                self.sign2id[token] = self.length
                self.id2sign[self.length] = token
                self.length += 1

        print(f"Vocabulary built with {self.length} tokens (min_freq={self.min_freq}).")

    def __len__(self):
        return self.length

    def save(self, path):
        """Saves the Vocab object to a file using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Vocabulary saved to: {path}")

    @staticmethod
    def load(path):
        """Loads a Vocab object from a file using pickle."""
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        print(f"Vocabulary loaded from: {path}")
        return vocab


def execute(data_dir="./processed_data", output_dir="./processed_data", min_token_frequency=1):
    """
    Executes the vocabulary building process and calculates max sequence length.

    Args:
        data_dir (str): Path to the directory containing raw CSV datasets.
        output_dir (str): Directory where the built vocabulary will be saved.
        min_token_frequency (int): Minimum frequency for a token to be included in the vocabulary.
    """
    print("Starting vocabulary building process...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    vocab_save_path = os.path.join(output_dir, "vocab.pkl")
    max_seq_len_file_path = os.path.join(output_dir, 'max_seq_len.txt')  # Added path for max_seq_len

    dataset_names = ["train", "val", "test"]
    all_formulas = []

    # Collect all formulas from all splits
    print(f"Collecting formulas from: {data_dir}")
    for name in dataset_names:
        csv_path = os.path.join(data_dir, f"{name}_processed.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found: {csv_path}. Skipping.")
            continue
        df = pd.read_csv(csv_path)
        # Convert formula column to string to handle potential non-string entries
        formulas_from_split = df["formula"].astype(str).tolist()
        all_formulas.extend(formulas_from_split)

    if not all_formulas:
        print("Error: No formulas found to build vocabulary. Check DATA_DIR and CSV file names.")
        return

    # Create and build the vocabulary
    vocab = Vocab(min_freq=min_token_frequency)
    vocab.build(all_formulas)
    print(f"Vocabulary built with {len(vocab)} tokens.")

    # Save the vocabulary
    vocab.save(vocab_save_path)

    # --- START OF ADDED MAX_SEQ_LEN LOGIC ---
    print("\nCalculating maximum sequence length...")

    max_seq_len = 0

    # Define the tokenization for length calculation, which must match how dataset.py does it (including SOS/EOS)
    def tokenize_latex_for_len_calc(formula):
        # This function should be IDENTICAL to tokenize_latex in dataset.py
        # It's defined here locally to avoid importing dataset.py and potential circular dependencies.
        if formula.startswith('"') and formula.endswith('"'):
            formula = formula[1:-1]
        formula = formula.strip()
        formula = re.sub(r'\s+', ' ', formula)
        tokens = [m.group(0) for m in latex_tokenizer_pattern.finditer(formula)]
        return ["<s>"] + tokens + ["</s>"]  # Add START/END tokens here

    # Use the vocabulary and the specific tokenization for length calculation
    for formula_str in tqdm(all_formulas, desc="Determining Max Sequence Length"):
        # Convert tokens to IDs to get sequence length (will use UNK_TOKEN for unknown)
        tokens_with_sos_eos = tokenize_latex_for_len_calc(formula_str)
        current_seq_ids = [vocab.sign2id.get(token, UNK_TOKEN) for token in tokens_with_sos_eos]

        if len(current_seq_ids) > max_seq_len:
            max_seq_len = len(current_seq_ids)

    # Add a small buffer for safety/future changes
    FINAL_MAX_SEQ_LEN = max_seq_len + 10
    with open(max_seq_len_file_path, 'w') as f:
        f.write(str(FINAL_MAX_SEQ_LEN))
    print(f"Calculated MAX_SEQ_LEN (with buffer): {FINAL_MAX_SEQ_LEN} and saved to {max_seq_len_file_path}")
    # --- END OF ADDED MAX_SEQ_LEN LOGIC ---

    # Example Usage (for verification, can be removed in final script)
    print("\n--- Vocabulary Example Usage ---")
    loaded_vocab = Vocab.load(vocab_save_path)
    print(f"Token for ID {20} (example): {loaded_vocab.id2sign.get(20, '<unk>')}")

    print("\nVocabulary building and max sequence length calculation complete!")


if __name__ == "__main__":
    execute(data_dir="./processed_data", output_dir="./processed_data", min_token_frequency=1)