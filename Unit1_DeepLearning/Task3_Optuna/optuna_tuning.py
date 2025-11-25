import os
import time
import json
import random
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------- Config ----------
OUTPUT_DIR = "./optuna_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# For tuning keep low epochs
EPOCHS_PER_TRIAL = 3
DEVICE = "cpu"
NUM_WORKERS = 0
PIN_MEMORY = False
SEED_BASE = 42

# ---------- Model ----------
class SimpleCNN(nn.Module):
    def __init__(self, n_filters=32, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, n_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(dropout)
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
        return self.fc2(x)

# ---------- Data (MNIST) ----------
def get_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(".", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(".", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    return train_loader, test_loader

# ---------- Training / Eval ----------
def train_one_epoch(model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
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

# ---------- Optuna objective ----------
def objective(trial):
    # sample hyperparams
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_filters = trial.suggest_categorical("n_filters", [16, 32, 48, 64])
    dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

    # deterministic-ish seed per trial
    trial_seed = SEED_BASE + trial.number
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(trial_seed)

    train_loader, test_loader = get_loaders(batch_size)
    model = SimpleCNN(n_filters=n_filters, dropout=dropout).to(DEVICE)

    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    start = time.time()
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS_PER_TRIAL + 1):
        tr_loss = train_one_epoch(model, optimizer, train_loader, DEVICE)
        val_acc, val_loss = eval_model(model, test_loader, DEVICE)
        # report intermediate value for pruning
        trial.report(val_acc, epoch)
        # prune if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    elapsed = time.time() - start

    # save per-trial short summary (optional)
    trial_info = {
        "trial": trial.number,
        "params": {"lr": lr, "batch_size": batch_size, "n_filters": n_filters, "dropout": dropout, "optimizer": optimizer_name},
        "val_acc": best_val_acc,
        "time_s": elapsed
    }
    # append to csv (thread-safe enough for single-process Optuna)
    csv_path = os.path.join(OUTPUT_DIR, "optuna_trials.csv")
    write_header = not os.path.exists(csv_path)
    line = json.dumps(trial_info)
    with open(csv_path, "a") as f:
        if write_header:
            f.write("json\n")
        f.write(line + "\n")

    return best_val_acc

# ---------- Main ----------
def main():
    study_path = os.path.join(OUTPUT_DIR, "optuna_study.db")
    storage = f"sqlite:///{study_path}"
    study = optuna.create_study(direction="maximize", storage=storage, study_name="mnist_cnn", load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1))
    N_TRIALS = 12

    start_all = time.time()
    study.optimize(objective, n_trials=N_TRIALS)
    total_time = time.time() - start_all

    # save summary
    summary = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "total_time_s": total_time
    }
    with open(os.path.join(OUTPUT_DIR, "optuna_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Study finished. Best value:", study.best_value)
    print("Best params:", study.best_params)
    print("Total tuning time (s):", total_time)

if __name__ == "__main__":
    main()
