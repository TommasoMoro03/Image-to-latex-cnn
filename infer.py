# infer.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
import pickle
import re

# Import your custom modules
from models.cnn_lstm_attention import ImageToLatexModel
# Import special token IDs and the VocabularyLoader class from dataset.py
from dataset import START_TOKEN_ID, END_TOKEN_ID
from build_vocab import Vocab


def infer_latex(
        image_path,
        model_load_path,
        encoder_dim,
        decoder_dim,
        embed_dim,
        dropout_prob,
        target_img_width,
        target_img_height,
        max_decoding_steps=500  # Max tokens to generate to prevent infinite loops
):
    """
    Performs inference on a single image to predict its LaTeX formula.

    Args:
        image_path (str): Path to the input image file.
        model_load_path (str): Path to the .pth file of the trained model state_dict.
        encoder_dim (int): Dimensionality of the encoder's output features.
        decoder_dim (int): Dimensionality of the decoder's LSTM hidden state.
        embed_dim (int): Dimensionality of token embeddings.
        dropout_prob (float): Dropout probability (used for model architecture).
        target_img_width (int): Target width for image preprocessing.
        target_img_height (int): Target height for image preprocessing.
        max_decoding_steps (int): Maximum number of tokens to generate during inference.
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

    # --- Load Vocabulary ---
    # Assuming vocab.pkl is in ./processed_data relative to where you run infer.py
    vocab_file = os.path.join("./processed_data", 'vocab.pkl')
    if not os.path.exists(vocab_file):
        print(f"Error: Vocabulary file not found at {vocab_file}. Please run build_vocab.py first.")
        return
    vocab_obj = Vocab.load(vocab_file)
    vocab_size = len(vocab_obj)

    # --- Load Max Sequence Length (for safety/consistency, though not directly used for padding here) ---
    max_seq_len_file = os.path.join("./processed_data", 'max_seq_len.txt')
    if not os.path.exists(max_seq_len_file):
        print(f"Warning: max_seq_len.txt not found at {max_seq_len_file}. Using a default max decoding steps.")
        max_model_seq_len = max_decoding_steps  # Fallback
    else:
        with open(max_seq_len_file, 'r') as f:
            max_model_seq_len = int(f.read().strip())

    # Use the minimum of max_decoding_steps and max_model_seq_len
    # to prevent excessively long or infinite generation
    max_tokens_to_generate = min(max_decoding_steps, max_model_seq_len)

    # --- Image Transformation ---
    image_transform = transforms.Compose([
        transforms.Resize((target_img_height, target_img_width)),  # Resize expects (height, width)
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5]) # Apply if you used this during training
    ])

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
        print(f"Error: Model not found at {model_load_path}. Please check path.")
        return

    print(f"Loading model state from {model_load_path}")
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # --- Load and Preprocess Input Image ---
    print(f"\nProcessing input image: {image_path}")
    if not os.path.exists(image_path):
        print(f"Error: Input image not found at {image_path}.")
        return

    try:
        image = Image.open(image_path).convert('L')  # Grayscale
        image_tensor = image_transform(image)  # Apply transformations
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension (1, C, H, W)
        image_tensor = image_tensor.to(device)
    except Exception as e:
        print(f"Error processing input image {image_path}: {e}")
        return

    # --- Inference (Greedy Decoding) ---
    print("Generating LaTeX sequence...")
    predicted_ids = []

    # Initialize decoder's hidden and cell states from encoder output
    with torch.no_grad():
        encoder_out = model.encoder(image_tensor)  # (1, num_pixels, encoder_dim)
        # Initialize decoder states with a dummy first token ID
        # The decoder's init_hidden_state method expects the full encoder_out
        h, c = model.decoder.init_hidden_state(encoder_out)

        # Start with the SOS token
        input_token = torch.tensor([START_TOKEN_ID], dtype=torch.long, device=device)  # (1,)

        for t in range(max_tokens_to_generate):
            # Pass one token at a time to the decoder
            # The decoder's forward method from cnn_lstm_attention.py expects (B, 1) for input_token_ids
            # so we need to unsqueeze it if it expects (B, Seq_len) as its 'input_token_ids' param
            # or pass the first token to kick off generation.

            # The decoder's forward pass is designed for teacher-forcing (full sequence at once).
            # For inference (greedy decoding), we need to call LSTMCell and Attention directly.

            # Get context vector from attention
            context, _ = model.decoder.attention(encoder_out, h)  # context: (1, encoder_dim)

            # Concatenate context vector with embedding of previous token (current input_token)
            embeddings = model.decoder.embedding(input_token)  # (1, embed_dim)
            lstm_input = torch.cat((embeddings, context), dim=1)  # (1, embed_dim + encoder_dim)

            # Pass through LSTM cell
            h, c = model.decoder.lstm(lstm_input, (h, c))  # h, c: (1, decoder_dim)

            # Apply gating and final classification layer
            gate = model.decoder.sigmoid(model.decoder.f_beta(h))  # (1, encoder_dim)
            gated_context = gate * context  # (1, encoder_dim)

            # Predict next token
            score = model.decoder.fc(torch.cat((h, gated_context), dim=1))  # (1, vocab_size)

            # Get the token with the highest probability (greedy approach)
            _, next_token_id = torch.max(score, dim=1)  # (1,)

            predicted_ids.append(next_token_id.item())

            # Stop if EOS token is predicted
            if next_token_id.item() == END_TOKEN_ID:
                break

            # Set the predicted token as the input for the next step
            input_token = next_token_id  # (1,)

    # --- Decode Predicted IDs to LaTeX String ---
    decoded_latex = []
    for token_id in predicted_ids:
        if token_id == END_TOKEN_ID:
            break  # Stop at EOS
        token = vocab_obj.id2sign.get(token_id, '<unk>')
        decoded_latex.append(token)

    # Join tokens to form the final LaTeX string
    final_latex_string = "".join(decoded_latex).replace(" ", " ")  # Normalize spaces after joining

    print("\n--- Inference Result ---")
    print(f"Input Image: {image_path}")
    print(f"Predicted LaTeX: {final_latex_string}")
    print("------------------------")

    return final_latex_string


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Image-to-LaTeX Inference")
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image file (e.g., ./processed_data/test_images/your_formula_padded.png).')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the .pth file of the trained model state_dict (e.g., ./saved_models/best_cnn_lstm_attention.pth).')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Dimensionality of encoder output features.')
    parser.add_argument('--decoder_dim', type=int, default=512, help='Dimensionality of decoder hidden state.')
    parser.add_argument('--embed_dim', type=int, default=256, help='Dimensionality of token embeddings.')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability (for model architecture).')
    parser.add_argument('--target_img_width', type=int, default=800, help='Target width for image preprocessing.')
    parser.add_argument('--target_img_height', type=int, default=160, help='Target height for image preprocessing.')
    parser.add_argument('--max_decoding_steps', type=int, default=500,
                        help='Maximum tokens to generate during inference.')

    args = parser.parse_args()

    # Run inference
    infer_latex(
        image_path=args.image_path,
        model_load_path=args.model_path,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        embed_dim=args.embed_dim,
        dropout_prob=args.dropout_prob,
        target_img_width=args.target_img_width,
        target_img_height=args.target_img_height,
        max_decoding_steps=args.max_decoding_steps
    )