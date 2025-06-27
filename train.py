import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 

# Import your custom modules
from dataset import get_dataloaders, PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID, UNK_TOKEN_ID
from build_vocab import Vocab 
# Import both model architectures (or conditionally import later)
from models.cnn_lstm_attention import ImageToLatexModel as CNNLSTMModel # Rename for clarity
from models.cnn_transformer_decoder import ImageToLatexTransformerModel as CNNTransformerModel # Rename for clarity


def save_checkpoint(state, ckpt_path="checkpoint_last.pt", best_ckpt_path="checkpoint_best.pt"):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(state, ckpt_path)
    if state.get("is_best", False):
        torch.save(state, best_ckpt_path)


def load_checkpoint(ckpt_path, device):
    if os.path.isfile(ckpt_path):
        print(f"Restarting from checkpoint {ckpt_path}")
        return torch.load(ckpt_path, map_location=device)
    return None


def predict_latex_sample(model, image_tensor, vocab_obj, device, max_decoding_steps=500, model_type="lstm"):
    """
    Performs greedy inference on a single image tensor.
    Added model_type to handle different inference logic for Transformer.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)

        predicted_ids = []
        
        encoder_out = model.encoder(image_tensor) # (1, num_pixels, encoder_dim)

        # Transformer inference is different from LSTM
        if model_type == "lstm":
            h, c = model.decoder.init_hidden_state(encoder_out)
            input_token = torch.tensor([START_TOKEN_ID], dtype=torch.long, device=device) # (1,)
            
            for _ in range(max_decoding_steps):
                context, _ = model.decoder.attention(encoder_out, h)
                embeddings = model.decoder.embedding(input_token)
                lstm_input = torch.cat((embeddings, context), dim=1)
                h, c = model.decoder.lstm(lstm_input, (h, c))
                gate = model.decoder.sigmoid(model.decoder.f_beta(h))
                gated_context = gate * context
                score = model.decoder.fc(torch.cat((h, gated_context), dim=1))
                _, next_token_id = torch.max(score, dim=1)
                
                predicted_ids.append(next_token_id.item())
                if next_token_id.item() == END_TOKEN_ID:
                    break
                input_token = next_token_id

        elif model_type == "transformer":
            # For Transformer, we generate one token at a time, feeding output as input.
            # We need to maintain the sequence built so far.
            # input_tokens_so_far starts with SOS token for the first prediction
            input_tokens_so_far = torch.tensor([[START_TOKEN_ID]], dtype=torch.long, device=device) # (1, 1)

            # Project encoder output if dimensions don't match embed_dim (d_model)
            # This logic should ideally be handled within the model's forward
            # but for inference, we'll ensure consistency
            encoder_out_projected = model.encoder_output_projection(encoder_out)

            for _ in range(max_decoding_steps):
                # Transformer Decoder expects (B, L_tgt) for tgt_token_ids
                # and needs masks
                tgt_len = input_tokens_so_far.size(1)
                tgt_mask = model.decoder._generate_square_subsequent_mask(tgt_len).to(device)
                
                # Transformer prediction
                # It will internally use self-attention on input_tokens_so_far
                # and cross-attention on encoder_out_projected
                logits_all_steps = model.decoder(
                    tgt_token_ids=input_tokens_so_far,
                    memory=encoder_out_projected,
                    tgt_mask=tgt_mask,
                    tgt_padding_mask=None, # Assuming no padding in this dynamically growing sequence
                    memory_padding_mask=None # No padding in encoder_out
                )
                
                # Get the last predicted token's logits
                # logits_all_steps shape: (B, L_tgt, vocab_size)
                # We need the logits for the *last* token in the sequence (at index tgt_len - 1)
                last_token_logits = logits_all_steps[:, -1, :] # (1, vocab_size)

                _, next_token_id = torch.max(last_token_logits, dim=1) # (1,)
                
                predicted_ids.append(next_token_id.item())

                if next_token_id.item() == END_TOKEN_ID:
                    break
                
                # Append the predicted token to the input sequence for the next step
                input_tokens_so_far = torch.cat([input_tokens_so_far, next_token_id.unsqueeze(0)], dim=1)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    decoded_latex = []
    for token_id in predicted_ids:
        if token_id == END_TOKEN_ID:
            break
        token = vocab_obj.id2sign.get(token_id, '<unk>')
        decoded_latex.append(token)
    
    return "".join(decoded_latex).replace(" ", " ")


def train_model(
        batch_size=16,
        num_epochs=10,
        learning_rate=1e-4,
        encoder_dim=512,
        decoder_dim=512, # For LSTM, this is hidden size. For Transformer, this might map to embed_dim/d_model.
        embed_dim=256,   # For LSTM, embedding size. For Transformer, also d_model.
        dropout_prob=0.5,
        target_img_width=800,
        target_img_height=160,
        num_workers=4,
        workdir="./saved_models",
        ckpt_name="cnn_lstm",
        patience=3,
        model_type="lstm", # NEW: Argument to select model type
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
):
    os.makedirs(workdir, exist_ok=True)
    ckpt_path_last = os.path.join(workdir, f"{ckpt_name}_last.pt")
    ckpt_path_best = os.path.join(workdir, f"{ckpt_name}_best.pt")
    metrics_path = os.path.join(workdir, f"{ckpt_name}_metrics.json")

    # ---------- device ----------
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    # ---------- data ----------
    train_loader, val_loader, _, vocab, max_seq = get_dataloaders(
        batch_size=batch_size, num_workers=num_workers,
        target_img_size=(target_img_width, target_img_height)
    )
    vocab_size = len(vocab)

    # --- Select a fixed sample for periodic inference display ---
    fixed_val_image = None
    fixed_val_target_formula = None
    if len(val_loader) > 0:
        first_batch = next(iter(val_loader))
        fixed_val_image = first_batch[0][0].unsqueeze(0)
        fixed_val_target_ids = first_batch[2][0]
        decoded_target_tokens = [
            vocab.id2sign.get(idx.item(), '<unk>') 
            for idx in fixed_val_target_ids 
            if idx.item() not in [PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID]
        ]
        fixed_val_target_formula = "".join(decoded_target_tokens).replace(" ", " ")
        
        print(f"\nMonitoring sample selected (from validation set):")
        print(f"  Actual LaTeX: {fixed_val_target_formula}")
    else:
        print("Warning: Validation loader is empty, cannot select monitoring sample.")

    # ---------- model ----------
    encoded_image_size = (target_img_height // 8, target_img_width // 8)
    
    # NEW: Dynamically choose model based on model_type argument
    if model_type == "lstm":
        model = CNNLSTMModel(vocab_size, embed_dim,
                               encoder_dim, decoder_dim,
                               encoded_image_size, dropout_prob).to(device)
    elif model_type == "transformer":
        # For Transformer, embed_dim serves as d_model.
        # encoder_dim should typically match embed_dim for cross-attention.
        # Ensure your Transformer model params are consistent.
        # Assuming nhead=8, num_decoder_layers=6, dim_feedforward=embed_dim*4 for Transformer
        model = CNNTransformerModel(vocab_size, embed_dim,
                                    encoder_dim,
                                    encoded_image_size, dropout_prob,
                                    nhead, num_decoder_layers, dim_feedforward=embed_dim*4,
                                    max_seq_len=max_seq).to(device) # Pass max_seq_len to Transformer
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'lstm' or 'transformer'.")

    print(f"Using model type: {model_type}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    # ---------- criterion / optim ----------
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch   = 0
    best_val_loss = float("inf")
    history       = []

    # ---------- resume ----------
    ckpt = load_checkpoint(ckpt_path_last, device)
    if ckpt:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        history       = ckpt.get("history", [])
        print(f"Restarted from epoch {start_epoch} with best_val_loss={best_val_loss:.4f}")

    # ---------- training ----------
    epochs_no_improve = 0
    for epoch in range(start_epoch, num_epochs):
        try:
            # ----- train -----
            model.train()
            running_loss = 0
            loop = tqdm(train_loader, desc=f"[{epoch+1}/{num_epochs}] Train")
            for imgs, inp, tgt in loop:
                imgs, inp, tgt = imgs.to(device), inp.to(device), tgt.to(device)
                optimizer.zero_grad()
                
                # NEW: Pass PAD_TOKEN_ID to Transformer model forward if model_type is transformer
                if model_type == "lstm":
                    logits = model(imgs, inp)
                elif model_type == "transformer":
                    logits = model(imgs, inp, PAD_TOKEN_ID) # Transformer needs pad_token_id for masks
                
                loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            avg_train = running_loss / len(train_loader)

            # ----- val -----
            model.eval()
            running_val = 0
            with torch.no_grad():
                loop = tqdm(val_loader, desc=f"[{epoch+1}/{num_epochs}] Val  ")
                for imgs, inp, tgt in loop:
                    imgs, inp, tgt = imgs.to(device), inp.to(device), tgt.to(device)
                    
                    # NEW: Pass PAD_TOKEN_ID to Transformer model forward during validation
                    if model_type == "lstm":
                        logits = model(imgs, inp)
                    elif model_type == "transformer":
                        logits = model(imgs, inp, PAD_TOKEN_ID) # Transformer needs pad_token_id for masks
                    
                    loss = criterion(logits.view(-1, vocab_size), tgt.view(-1))
                    running_val += loss.item()
                    loop.set_postfix(val_loss=loss.item())
            avg_val = running_val / len(val_loader)

            print(f"Epoch {epoch+1}: train={avg_train:.4f}  val={avg_val:.4f}")

            # ----- history -----
            history.append({"epoch": epoch+1, "train_loss": avg_train, "val_loss": avg_val})
            with open(metrics_path, "w") as f:
                json.dump(history, f, indent=2)

            # ----- checkpoint -----
            is_best = avg_val < best_val_loss
            if is_best:
                best_val_loss = avg_val
                epochs_no_improve = 0
                print("New best model saved")
            else:
                epochs_no_improve += 1
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
                "is_best": is_best,
            }, ckpt_path_last, ckpt_path_best)

            # --- Display test sample prediction ---
            if fixed_val_image is not None:
                print("\n--- Monitoring Sample Prediction ---")
                # Pass model_type to predict_latex_sample
                predicted_latex = predict_latex_sample(model, fixed_val_image, vocab, device, model_type=model_type)
                print(f"Actual: {fixed_val_target_formula}")
                print(f"Predicted: {predicted_latex}")
                print("----------------------------------\n")

            # ----- early stop -----
            if epochs_no_improve >= patience:
                print(f"Early stopping: {epochs_no_improve} epochs without improvement")
                break

        except (KeyboardInterrupt, ConnectionError, RuntimeError) as e:
            print(f"\n⚠️  Interruption ({type(e).__name__}): save checkpoint before exiting...")
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "history": history,
                "is_best": False,
            }, ckpt_path_last, ckpt_path_best)
            raise e
    print("Training finished!")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Image-to-LaTeX Model")
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer.')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Dimensionality of encoder output features.')
    # For Transformer, decoder_dim and embed_dim often become the same parameter (d_model).
    # You might consider unifying them or carefully setting.
    parser.add_argument('--decoder_dim', type=int, default=512, help='Dimensionality of decoder hidden state (LSTM) or d_model (Transformer).')
    parser.add_argument('--embed_dim', type=int, default=256, help='Dimensionality of token embeddings (LSTM) or d_model (Transformer).')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability.')
    parser.add_argument('--target_img_width', type=int, default=800, help='Target width for processed images.')
    parser.add_argument('--target_img_height', type=int, default=160, help='Target height for processed images.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument("--workdir", type=str, default="./saved_models")
    parser.add_argument("--ckpt_name", type=str, default="cnn_lstm", help="Base name for checkpoint files (e.g., cnn_lstm or cnn_transformer).")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"],
                        help="Type of decoder model to use: 'lstm' or 'transformer'.") # NEW ARGUMENT
    
    # Transformer-specific arguments (provide defaults)
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads in Transformer.')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='Number of decoder layers in Transformer.')
    parser.add_argument('--dim_feedforward', type=int, default=2048, help='Dimension of feedforward network in Transformer.')


    args = parser.parse_args()

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
        workdir=args.workdir,
        ckpt_name=args.ckpt_name,
        patience=args.patience,
        model_type=args.model_type, # Pass the new argument
        nhead=args.nhead, # Pass Transformer-specific arguments
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward
    )
