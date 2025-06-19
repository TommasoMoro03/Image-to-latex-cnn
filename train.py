import json
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


def save_checkpoint(state, ckpt_path="checkpoint_last.pt", best_ckpt_path="checkpoint_best.pt"):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(state, ckpt_path)
    # save separately the best one
    if state.get("is_best", False):
        torch.save(state, best_ckpt_path)

def load_checkpoint(ckpt_path, device):
    if os.path.isfile(ckpt_path):
        print(f"Restarting from checkpoint {ckpt_path}")
        return torch.load(ckpt_path, map_location=device)
    return None

def train_model(
        batch_size=16,
        num_epochs=10,
        learning_rate=1e-4,
        encoder_dim=512,
        decoder_dim=512,
        embed_dim=256,
        dropout_prob=0.5,
        target_img_width=800,
        target_img_height=160,
        num_workers=4,
        workdir="./saved_models",
        ckpt_name="cnn_lstm",
        patience=3,
):
    os.makedirs(workdir, exist_ok=True)
    ckpt_path_last  = os.path.join(workdir, f"{ckpt_name}_last.pt")
    ckpt_path_best  = os.path.join(workdir, f"{ckpt_name}_best.pt")
    metrics_path    = os.path.join(workdir, f"{ckpt_name}_metrics.json")

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

    # ---------- model ----------
    encoded_image_size = (target_img_height // 8, target_img_width // 8)
    model = ImageToLatexModel(vocab_size, embed_dim,
                              encoder_dim, decoder_dim,
                              encoded_image_size, dropout_prob).to(device)

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
                logits = model(imgs, inp)
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
                    logits = model(imgs, inp)
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
    parser.add_argument('--decoder_dim', type=int, default=512, help='Dimensionality of decoder hidden state.')
    parser.add_argument('--embed_dim', type=int, default=256, help='Dimensionality of token embeddings.')
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='Dropout probability.')
    parser.add_argument('--target_img_width', type=int, default=800, help='Target width for processed images.')
    parser.add_argument('--target_img_height', type=int, default=160, help='Target height for processed images.')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for data loading.')
    parser.add_argument('--model_save_dir', type=str, default='./saved_models', help='Directory to save models.')
    parser.add_argument('--model_name', type=str, default='best_cnn_lstm_attention.pth',
                        help='Filename for the saved model.')
    parser.add_argument("--workdir", type=str, default="./saved_models")
    parser.add_argument("--ckpt_name", type=str, default="cnn_lstm")
    parser.add_argument("--patience", type=int, default=3)

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
        workdir=args.workdir,
        ckpt_name=args.ckpt_name,
        patience=args.patience
    )