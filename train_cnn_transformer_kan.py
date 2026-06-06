import argparse
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from dataloader_updated import create_dataloaders
from models.cnn_transformer_kan import CNNTransformerKAN



# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 500
BATCH_SIZE = 8
LR = 1e-4

CNN_OUT_DIM = 64
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.4

WANDB_PROJECT = "e-nose-odor-classification"
RUN_NAME = "CNN_Transformer___min_max_norm"
# =========================================


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct1 = 0
    correct5 = 0
    total = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        # Top-1
        preds1 = logits.argmax(dim=1)
        correct1 += (preds1 == y).sum().item()

        # Top-5
        k = min(5, logits.size(1))
        top5 = logits.topk(k, dim=1).indices
        correct5 += top5.eq(y.view(-1, 1)).any(dim=1).sum().item()

        total += y.size(0)

    return total_loss / len(loader), correct1 / total, correct5 / total

@torch.no_grad()
def evaluate_oils_only(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    dataset = loader.dataset

    for batch_indices, (x, y) in zip(loader.batch_sampler, loader):
        # batch_indices are TRUE dataset indices
        batch_samples = [dataset.samples[i] for i in batch_indices]

        # Derive coarse class from the stored path so both dataset versions work
        mask = [
            (sample[0].parents[2].name if len(sample) >= 1 else None) == "oils"
            for sample in batch_samples
        ]

        if not any(mask):
            continue

        mask = torch.tensor(mask, dtype=torch.bool)

        x = x[mask].to(DEVICE)
        y = y[mask].to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        batch_n = y.size(0)
        total += batch_n
        # loss is averaged over the (masked) batch; weight by batch size
        total_loss += loss.item() * batch_n

    if total == 0:
        return None, None

    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")
    args = parser.parse_args()
    use_wandb = not args.no_wandb

    if use_wandb:
        import wandb
        wandb.init(
            project=WANDB_PROJECT,
            name=RUN_NAME,
            config={
                "model": "CNN_Transformer",
                "cnn_out_dim": CNN_OUT_DIM,
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "learning_rate": LR,
                "epochs": NUM_EPOCHS
            }
        )

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )
    # Add this here:
    import json
    with open("label_map.json", "w") as f:
        json.dump(train_loader.dataset.label_map, f)

    num_classes = len(train_loader.dataset.label_map)

    model = CNNTransformerKAN(
        num_classes=num_classes,
        cnn_out_dim=CNN_OUT_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        #val_loss, val_acc = evaluate(model, val_loader, criterion)
        val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion)


        #oils_loss, oils_acc = evaluate_oils_only(model, val_loader, criterion)

        #if oils_acc is not None:
        #    print(f"Oils-only Val Acc: {oils_acc:.4f}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_top1": val_top1,
                "val_top5": val_top5
            })

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Top-1: {val_top1:.4f} | Val Top-5: {val_top5:.4f}")

    torch.save(model.state_dict(), "cnn_transformer_final.pt")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
