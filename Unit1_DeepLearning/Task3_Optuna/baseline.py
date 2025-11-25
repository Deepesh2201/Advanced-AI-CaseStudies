import time
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------- CONFIG ----------
SEED = 40
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
OUTPUT_DIR = "./baseline_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- REPRO ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- MODEL ----------
class SimpleCNN(nn.Module):
    def __init__(self, n_filters=32, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(dropout)
        # after two pools, image size 28 -> 14 -> 7; if n_filters*2 channels:
        self.fc1 = nn.Linear((n_filters*2) * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transformer & Dataset MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(".", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(".", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if DEVICE=="cuda" else False)
test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=0, pin_memory=True if DEVICE=="cuda" else False)

# ---------- TRAIN / EVAL ----------
def train_one_epoch(model, opt, loader, device):
    model.train()
    total_loss = 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y, reduction="sum")
            loss_sum += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct/total, loss_sum/total

# ---------- RUN ----------
model = SimpleCNN(n_filters=32, dropout=0.25).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

start_time = time.time()
for epoch in range(1, EPOCHS+1):
    tr_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
    val_acc, val_loss = eval_model(model, test_loader, DEVICE)
    print(f"Epoch {epoch}/{EPOCHS} — train_loss: {tr_loss:.4f} — test_loss: {val_loss:.4f} — test_acc: {val_acc:.4f}")

total_time = time.time() - start_time
print("Baseline training time (s):", total_time)
print("Final test accuracy:", val_acc)

# save baseline metrics
metrics = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "epochs": EPOCHS,
    "device": DEVICE,
    "train_time_s": total_time,
    "test_accuracy": float(val_acc),
    "test_loss": float(val_loss)
}
with open(os.path.join(OUTPUT_DIR, "baseline_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# saved model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "baseline_model.pth"))
