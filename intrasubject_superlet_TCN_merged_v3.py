# intrasubject_superlet_TCN_merged_v3.py
import os
import math
import random
from functools import partial
from multiprocessing import Pool, cpu_count
from collections import Counter  # for class weights

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
FS = 100
FILES = ["/content/S1_A1_E1.mat", "/content/S1_A1_E2.mat", "/content/S1_A1_E3.mat"]

CACHE_FILE = "superlet_features_subject1_400ms_30freqs.npz"

N_PROCS = max(1, cpu_count() - 1)
BATCH_SIZE = 32
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUGMENT = True
EPOCHS = 100
LR = 5e-4
PATIENCE = 20  # more patience

# ---------------------------
# SUPERLET
# ---------------------------
def morlet_custom(M, w=6.0):
    t = np.linspace(-1, 1, M)
    a = w * math.pi
    wavelet = np.exp(1j * a * t) * np.exp(-(t**2) * (w**2))
    return wavelet.astype(np.complex64)

def superlet_transform(signal, fs, freqs, base_cycles=3, super_levels=3):
    n = len(signal)
    output = np.zeros((len(freqs), n), dtype=np.float32)
    for fi, f in enumerate(freqs):
        product = None
        for level in range(super_levels):
            cycles = base_cycles + level
            M = max(int(2 * cycles * fs / f), 4)
            wavelet = morlet_custom(M, w=cycles)
            conv = np.abs(np.convolve(signal, wavelet, mode='same'))
            if len(conv) != n:
                conv = np.interp(
                    np.linspace(0, len(conv)-1, n),
                    np.arange(len(conv)),
                    conv
                )
            product = conv if product is None else product * conv
        output[fi, :] = product ** (1.0 / super_levels)
    return output

def extract_superlet_features_single(trial, fs=FS, freqs=None, base_cycles=3, super_levels=3):
    if freqs is None:
        freqs = np.linspace(10, 150, 30)  # 30 frequencies
    C = trial.shape[0]
    T = trial.shape[1]
    out = np.zeros((C, len(freqs), T), dtype=np.float32)
    for ch in range(C):
        out[ch] = superlet_transform(
            trial[ch, :], fs, freqs,
            base_cycles=base_cycles,
            super_levels=super_levels
        )
    return out

# ---------------------------
# BUILD TRIALS FROM .mat FILES
# ---------------------------
def build_trials_from_mat_files(files, fs=FS, win_samples=None, step=None):
    all_emg = []
    all_labels = []
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f".mat file not found: {f}")
        mat = sio.loadmat(f)
        emg = mat["emg"].T              # (channels, samples)
        labels = mat["restimulus"].flatten()
        all_emg.append(emg)
        all_labels.append(labels)

    emg_concat = np.hstack(all_emg)
    labels_concat = np.hstack(all_labels)

    # ~400 ms window, ~200 ms step
    if win_samples is None:
        win_samples = int(0.4 * fs)
    if step is None:
        step = int(0.2 * fs)

    X_trials = []
    Y_trials = []
    total = emg_concat.shape[1]
    for start in range(0, total - win_samples + 1, step):
        seg = emg_concat[:, start:start+win_samples]
        lab = int(np.bincount(labels_concat[start:start+win_samples]).argmax())
        if lab == 0:
            continue
        X_trials.append(seg)
        Y_trials.append(lab)

    return np.array(X_trials), np.array(Y_trials)

# ---------------------------
# PARALLEL FEATURE EXTRACTION + CACHE
# ---------------------------
def _extract_for_pool(trial, freqs):
    return extract_superlet_features_single(trial, freqs=freqs)

def compute_and_cache_features(X_trials, Y_trials, cache_file=CACHE_FILE,
                               n_procs=N_PROCS, freqs=None):
    if os.path.exists(cache_file):
        print("Loading cache:", cache_file)
        d = np.load(cache_file, allow_pickle=False)
        if "X_feat" in d:
            X_feat = d["X_feat"]
            Y = d["Y"]
        elif "X" in d:
            X_feat = d["X"]
            Y = d["y"] if "y" in d else d["Y"]
        else:
            keys = list(d.keys())
            X_feat = d[keys[0]]
            Y = d[keys[1]] if len(keys) > 1 else Y_trials
        return X_feat, Y

    if freqs is None:
        freqs = np.linspace(10, 150, 30)
    print(f"Computing Superlet features with {n_procs} processes...")
    with Pool(processes=n_procs) as pool:
        results = pool.map(partial(_extract_for_pool, freqs=freqs), list(X_trials))
    X_feat = np.stack(results, axis=0)  # (N, C, F, T)
    np.savez_compressed(cache_file, X_feat=X_feat, Y=Y_trials)
    print("Saved cache:", cache_file)
    return X_feat, Y_trials

# ---------------------------
# AUGMENT (Superlet maps) – STRONGER
# ---------------------------
def augment_superlet_map(
    x,
    noise_std=0.03,
    max_shift=4,
    scale_low=0.95,
    scale_high=1.05,
    freq_mask_prob=0.5,
    max_freq_masks=2,
    max_freq_width=3
):
    """
    x: (C, F, T)
    """
    x = x.copy()

    # Gaussian noise
    x += np.random.randn(*x.shape) * noise_std

    # Global amplitude scaling
    scale = random.uniform(scale_low, scale_high)
    x *= scale

    # Temporal shift
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        pad = np.zeros((x.shape[0], x.shape[1], shift), dtype=x.dtype)
        x = np.concatenate([pad, x[:, :, :-shift]], axis=2)
    elif shift < 0:
        pad = np.zeros((x.shape[0], x.shape[1], -shift), dtype=x.dtype)
        x = np.concatenate([x[:, :, -shift:], pad], axis=2)

    # Frequency masking (SpecAugment-like)
    if random.random() < freq_mask_prob:
        F = x.shape[1]
        for _ in range(random.randint(1, max_freq_masks)):
            f0 = random.randint(0, F - 1)
            w = random.randint(1, max_freq_width)
            f1 = min(F, f0 + w)
            x[:, f0:f1, :] = 0.0

    return x

# ---------------------------
# DATASET
# ---------------------------
class SuperletDatasetFromFeat(Dataset):
    def __init__(self, X_feat, Y, augment=False):
        self.X = X_feat   # (N, C, F, T)
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = augment_superlet_map(x)
        C, F, T = x.shape
        x1d = x.reshape(C * F, T)
        return torch.tensor(x1d, dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.long)

# ---------------------------
# TCN + SE ATTENTION (with more dropout)
# ---------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation for channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T)
        w = x.mean(dim=2)          # (B, C)
        w = self.fc(w)             # (B, C)
        w = w.unsqueeze(-1)        # (B, C, 1)
        return x * w

class TCNNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(input_channels, 64, kernel_size=3, dilation=1, dropout=0.25),
            TemporalBlock(64, 128, kernel_size=3, dilation=2, dropout=0.30),
            TemporalBlock(128, 128, kernel_size=3, dilation=4, dropout=0.35),
        )

        self.se = SEBlock(128, reduction=16)

        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.global_max = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.tcn(x)
        x = self.se(x)
        avg = self.global_avg(x).squeeze(-1)
        mx = self.global_max(x).squeeze(-1)
        x = torch.cat([avg, mx], dim=1)
        return self.fc(x)

# ---------------------------
# TRAIN FUNCTION
# ---------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR,
                device="cpu", class_weights=None):
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.03)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    patience = PATIENCE
    patience_counter = 0

    train_losses, val_losses = [], []
    train_acc_list, val_acc_list = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * Xb.size(0)
            correct += (preds.argmax(dim=1) == yb).sum().item()

        train_acc = correct / len(train_loader.dataset)
        train_losses.append(train_loss / len(train_loader.dataset))
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * Xb.size(0)
                correct += (preds.argmax(dim=1) == yb).sum().item()

        val_acc = correct / len(val_loader.dataset)
        val_losses.append(val_loss / len(val_loader.dataset))
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch}/{epochs} | Train Loss {train_losses[-1]:.4f} Acc {train_acc*100:.2f}% | "
              f"Val Loss {val_losses[-1]:.4f} Acc {val_acc*100:.2f}%")

        scheduler.step()

        # Checkpoint
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), "best_tcn_superlet.pt")
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("⛔ Early stopping activated.")
            break

    return train_losses, val_losses, train_acc_list, val_acc_list

# ---------------------------
# MAIN
# ---------------------------
def main():
    print("Loading .mat data and generating trials...")
    freqs = np.linspace(10, 150, 30)

    X_trials, Y_trials = build_trials_from_mat_files(
        FILES,
        fs=FS,
        win_samples=int(0.4 * FS),
        step=int(0.2 * FS)
    )
    print("Total trials:", len(X_trials))

    X_feat, Y = compute_and_cache_features(
        X_trials, Y_trials,
        cache_file=CACHE_FILE,
        n_procs=N_PROCS,
        freqs=freqs
    )
    print("X_feat shape (N, C, F, T):", X_feat.shape)

    # Channel-frequency normalization
    mean = X_feat.mean(axis=(0, 3), keepdims=True)
    std = X_feat.std(axis=(0, 3), keepdims=True) + 1e-6
    X_feat = (X_feat - mean) / std

    # Map labels to 0..(n_classes-1)
    unique = np.unique(Y)
    label_map = {lab: i for i, lab in enumerate(unique)}
    Y_mapped = np.array([label_map[y] for y in Y], dtype=np.int64)
    n_classes = len(unique)

    # Class weights
    counts = Counter(Y_mapped.tolist())
    total = len(Y_mapped)
    class_weights_np = np.array([total / counts[i] for i in range(n_classes)], dtype=np.float32)
    class_weights = torch.tensor(class_weights_np)

    # Reshape for TCN: (N, C*F, T)
    N, C, F, T = X_feat.shape
    X_reshaped = X_feat.reshape(N, C * F, T)
    print("Final shape for model (N, C*F, T):", X_reshaped.shape)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_reshaped,
        Y_mapped,
        test_size=0.2,
        shuffle=True,
        stratify=Y_mapped
    )

    train_ds = SuperletDatasetFromFeat(
        X_train.reshape(-1, C, F, T),
        y_train,
        augment=AUGMENT
    )
    val_ds = SuperletDatasetFromFeat(
        X_val.reshape(-1, C, F, T),
        y_val,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    device = DEVICE
    model = TCNNet(input_channels=C * F, num_classes=n_classes).to(device)

    # Training
    train_losses, val_losses, train_acc_list, val_acc_list = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device,
        class_weights=class_weights
    )

    # Plots
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss during training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_list, label="Train Accuracy")
    plt.plot(val_acc_list, label="Val Accuracy")
    plt.title("Accuracy during training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.show()

    # Confusion matrix on val set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            preds = model(Xb)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=[f"Class {i}" for i in range(n_classes)])
    disp.plot(cmap="Blues")
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
