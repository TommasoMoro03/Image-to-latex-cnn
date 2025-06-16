# evaluate.py

import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm

# Import your custom modules
from dataset import get_dataloaders, START_TOKEN_ID, PAD_TOKEN_ID, END_TOKEN_ID
from models.cnn_lstm_attention import ImageToLatexModel


def evaluate_model(
        batch_size,
        encoder_dim,
        decoder_dim,
        embed_dim,
        dropout_prob,
        target_img_width,
        target_img_height,
        num_workers,
        model_load_path
):
    """
    Evaluates a trained Image-to-LaTeX model on the test dataset.

    Args:
        batch_size (int): Number of samples per batch.
        encoder_dim (int): Dimensionality of the encoder's output features.
        decoder_dim (int): Dimensionality of the decoder's LSTM hidden state.
        embed_dim (int): Dimensionality of token embeddings.
        dropout_prob (float): Dropout probability for regularization.
        target_img_width (int): Target width for image preprocessing.
        target_img_height (int): Target height for image preprocessing.
        num_workers (int): Number of subprocesses for data loading.
        model_load_path (str): Path to the saved model state_dict.
    """
    # --- Device Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU available, falling back to CPU.")

    # --- Data Loading ---
    print("\nSetting up DataLoaders (loading test set)...")
    # We only need the test_loader for evaluation
    _, _, test_loader, vocab_obj, max_seq_len = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        target_img_size=(target_img_width, target_img_height)
    )
    vocab_size = len(vocab_obj)
    print(f"Vocabulary size: {vocab_size}, Max sequence length: {max_seq_len}")

    # --- Model Initialization ---
    print("\nInitializing model...")
    encoded_image_size = (
        target_img_height // 8,
        target_img_width // 8
    )

    model = ImageToLatexModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        encoded_image_size=encoded_image_size,
        dropout_prob=dropout_prob
    )

    # Load trained model weights
    if not os.path.exists(model_load_path):
        print(f"Error: Model not found at {model_load_path}. Please train the model first.")
        return

    print(f"Loading model state from {model_load_path}")
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # --- Loss Function (for calculating test loss) ---
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    # --- Evaluation Loop ---
    print("\nStarting evaluation...")
    total_test_loss = 0
    correct_sequences_exact_match = 0
    total_sequences = 0

    # Disable gradient calculations for evaluation
    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Evaluating Model")
        for images, input_ids, target_ids in test_loop:
            images, input_ids, target_ids = images.to(device), input_ids.to(device), target_ids.to(device)

            output_logits = model(images, input_ids)

            # Calculate loss
            loss = criterion(
                output_logits.view(-1, vocab_size),
                target_ids.view(-1)
            )
            total_test_loss += loss.item()

            # Calculate Exact Match Accuracy
            # Get the predicted token IDs for each sequence
            # output_logits shape: (batch_size, seq_len-1, vocab_size)
            _, predicted_ids = torch.max(output_logits, dim=2)  # predicted_ids shape: (batch_size, seq_len-1)

            # Compare predicted_ids with target_ids
            # We need to consider padding and stop at the first <EOS> token
            for i in range(batch_size):
                # Find the end of the actual sequence in target_ids
                # If END_TOKEN_ID is not present (e.g., due to truncation or error),
                # consider the sequence up to the first PAD_TOKEN_ID or max_seq_len-1
                target_sequence_len = (target_ids[i] == END_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(target_sequence_len) > 0:
                    seq_len = target_sequence_len[0].item() + 1  # Include EOS
                else:
                    seq_len = max_seq_len - 1  # Use full length if EOS not found (e.g., truncated)

                # Get the relevant parts of the sequences, excluding padding
                # Exclude padding, and any tokens beyond the actual sequence length
                actual_target = target_ids[i, :seq_len]
                actual_prediction = predicted_ids[i, :seq_len]  # Use same length for prediction

                if torch.equal(actual_target, actual_prediction):
                    correct_sequences_exact_match += 1
                total_sequences += 1

            test_loop.set_postfix(loss=loss.item(), acc=(
                                                                    correct_sequences_exact_match / total_sequences) * 100 if total_sequences > 0 else 0)

    avg_test_loss = total_test_loss / len(test_loader)
    exact_match_accuracy = (correct_sequences_exact_match / total_sequences) * 100 if total_sequences > 0 else 0

    print(f"\n--- Evaluation Results ---")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.2f}% ({correct_sequences_exact_match}/{total_sequences})")
    print(f"--------------------------")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Image-to-LaTeX Model")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Dimensionality of encoder output features.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='Dimensionality of decoder hidden state.')
    parser.add_argument('--embed_dim', type=int, default=256, help='Dimensionality of token embeddings.')
    parser.add_argument('--dropout_prob', type=float, default=0.5,
                        help='Dropout probability (used for model architecture).')
    parser.add_argument('--target_img_width', type=int, default=800, help='Target width for processed images.')
    parser.add_argument('--target_img_height', type=int, default=160, help='Target height for processed images.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the .pth file of the trained model state_dict.')

    args = parser.parse_args()

    # Run the evaluation
    evaluate_model(
        batch_size=args.batch_size,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        embed_dim=args.embed_dim,
        dropout_prob=args.dropout_prob,
        target_img_width=args.target_img_width,
        target_img_height=args.target_img_height,
        num_workers=args.num_workers,
        model_load_path=args.model_path
    )