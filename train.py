import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse

# Import your custom modules
from dataset import get_dataloaders, PAD_TOKEN_ID
from build_vocab import Vocab
from models.cnn_lstm_attention import ImageToLatexModel


def train_model(
        batch_size,
        num_epochs,
        learning_rate,
        encoder_dim,
        decoder_dim,
        embed_dim,
        dropout_prob,
        target_img_width,
        target_img_height,
        num_workers,
        model_save_path,
        log_interval=100
):
    """
    Orchestrates the training and validation process for the Image-to-LaTeX model.

    Args:
        batch_size (int): Number of samples per batch.
        num_epochs (int): Total number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        encoder_dim (int): Dimensionality of the encoder's output features.
        decoder_dim (int): Dimensionality of the decoder's LSTM hidden state.
        embed_dim (int): Dimensionality of token embeddings.
        dropout_prob (float): Dropout probability for regularization.
        target_img_width (int): Target width for image preprocessing.
        target_img_height (int): Target height for image preprocessing.
        num_workers (int): Number of subprocesses for data loading.
        model_save_path (str): Path to save the best performing model.
        log_interval (int): How often to print training loss (in batches).
    """
    # --- Device Setup ---
    # Check for MPS (Apple Silicon GPU) first, then CUDA (NVIDIA GPU), then fall back to CPU
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
    print("\nSetting up DataLoaders...")
    train_loader, valid_loader, _, vocab_obj, max_seq_len = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        target_img_size=(target_img_width, target_img_height)
    )
    vocab_size = len(vocab_obj)
    print(f"Vocabulary size: {vocab_size}, Max sequence length: {max_seq_len}")

    # --- Model Initialization ---
    print("\nInitializing model...")
    # The CNNEncoder's output feature map dimensions.
    # If input is 160x800 and CNN downsamples by 8x, output is 20x100.
    # This should match what's hardcoded/designed in models/cnn_lstm_attention.py
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
    model.to(device)

    # --- Loss Function and Optimizer ---
    # CrossEntropyLoss expects logits (raw scores) and target IDs.
    # PAD_TOKEN_ID should be ignored in loss calculation.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training Loop ---
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        # Use tqdm for progress bar
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train)")
        for batch_idx, (images, input_ids, target_ids) in enumerate(train_loop):
            images, input_ids, target_ids = images.to(device), input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()

            # Forward pass: model outputs logits for each token
            # output_logits shape: (batch_size, seq_len-1, vocab_size)
            output_logits = model(images, input_ids)

            # Reshape logits and targets for CrossEntropyLoss
            # CrossEntropyLoss expects (N, C) where N is batch size, C is number of classes
            # Here, N = (batch_size * (seq_len-1)), C = vocab_size
            loss = criterion(
                output_logits.view(-1, vocab_size),  # Flatten sequence and batch dimensions
                target_ids.view(-1)  # Flatten target IDs
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Update tqdm postfix with current loss
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0

        # Disable gradient calculations for validation
        with torch.no_grad():
            val_loop = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Valid)")
            for images, input_ids, target_ids in val_loop:
                images, input_ids, target_ids = images.to(device), input_ids.to(device), target_ids.to(device)

                output_logits = model(images, input_ids)

                loss = criterion(
                    output_logits.view(-1, vocab_size),
                    target_ids.view(-1)
                )
                total_val_loss += loss.item()
                val_loop.set_postfix(val_loss=loss.item())

        avg_val_loss = total_val_loss / len(valid_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Avg Validation Loss: {avg_val_loss:.4f}")

        # --- Model Saving ---
        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), model_save_path)
            best_val_loss = avg_val_loss
        else:
            print(f"Validation loss did not improve ({avg_val_loss:.4f} vs {best_val_loss:.4f}).")

    print("\nTraining complete!")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image-to-LaTeX Model")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer.')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Dimensionality of encoder output features.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='Dimensionality of decoder hidden state.')
    parser.add_argument('--embed_dim', type=int, default=256, help='Dimensionality of token embeddings.')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability.')
    parser.add_argument('--target_img_width', type=int, default=800, help='Target width for processed images.')
    parser.add_argument('--target_img_height', type=int, default=160, help='Target height for processed images.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help='Directory to save models.')
    parser.add_argument('--model_name', type=str, default='best_cnn_lstm_attention.pth',
                        help='Filename for the saved model.')

    args = parser.parse_args()

    # Create the model save directory if it doesn't exist
    os.makedirs(args.model_save_dir, exist_ok=True)
    full_model_save_path = str(os.path.join(args.model_save_dir, args.model_name))

    # Run the training
    train_model(
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        embed_dim=args.embed_dim,
        dropout_prob=args.dropout_prob,
        target_img_width=args.target_img_width,
        target_img_height=args.target_img_height,
        num_workers=args.num_workers,
        model_save_path=full_model_save_path
    )