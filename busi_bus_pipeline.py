# BUSI-BUS classification pipeline (Kaggle).
# RUN_MODE:
# - "manual_seed": train/evaluate a single seed.
# - "all_seeds": train/evaluate all seeds and aggregate final statistics.
# - "final_statistics_only": skip training, read existing results and aggregate statistics.

import os
import sys
import subprocess
import time

# CONFIG
DATASET_DIR = "/kaggle/input/datasets/harolmp/busi-bus/Dataset"
RESULTS_BASE = "/kaggle/working/results"

IMG_SIZE = 224
EPOCHS = 30

# 0 avoids extra variance from multiprocessing.
NUM_WORKERS = 0

# Keep fixed across runs so train/val/test splits match for paired tests.
SPLIT_SEED = 42

# Seeds used for the >=5 repeated runs reported in the paper.
RUN_SEEDS = [42, 123, 2024, 2025, 777]

# Used only in "manual_seed" mode.
RUN_SEED = 123
SEED = RUN_SEED

RUN_MODE = "all_seeds"

# Optuna HPO
HPO_N_TRIALS = 25
REOPTIMIZE_PER_SEED = False

# --- XAI (Pointing Game) term added to the Optuna objective ---
# Rewards models whose Integrated Gradients attribution falls inside the
# lesion mask. Evaluated on a small fixed reference set with fewer IG steps
# to keep HPO cost low.
XAI_HPO_ENABLE = True
XAI_HPO_N_SAMPLES = 6      # reference images used per evaluation
XAI_HPO_IG_STEPS = 20      # IG steps during HPO (fewer than the 50 used for final eval)
XAI_HPO_WEIGHT = 0.15      # weight of the XAI term in the Optuna score (0-1)

# Bootstrap / 95% CI
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 2026
ALPHA = 0.05

# XAI / Pointing Game
RUN_XAI_POINTING_GAME = True
SAVE_EXAMPLE_HEATMAPS = True
SAVE_ALL_HEATMAPS = False
XAI_ATTRIBUTION_THRESHOLD = 0.5

# ImageNet pretraining; may fail without internet/cache on Kaggle.
PRETRAINED = True

EXPERIMENT_ID = "busi_bus_results"
RESULTS_DIR = os.path.join(RESULTS_BASE, EXPERIMENT_ID)

# Set False if all dependencies are already installed.
INSTALL_MISSING_PACKAGES = True

# Set before importing torch.
os.environ["PYTHONHASHSEED"] = str(RUN_SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if INSTALL_MISSING_PACKAGES:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "timm",
            "scikit-learn",
            "optuna",
            "opencv-python-headless",
            "huggingface_hub",
            "scipy",
        ],
        check=False,
    )

# IMPORTS
import json
import platform
import random
from itertools import combinations

import cv2
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import sklearn
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from optuna.pruners import MedianPruner
from PIL import Image
from scipy.stats import binomtest, norm
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

import matplotlib.pyplot as plt

# GLOBAL SETTINGS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kept False to avoid mixed-precision variance.
USE_AMP = False

CLASS_NAMES = ["benign", "malignant", "normal"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: cls_name for cls_name, i in CLASS_TO_IDX.items()}

PATIENCE = 5
MIN_DELTA = 1e-4

os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            print(f"Could not enable strict determinism: {exc}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_dataloader(dataset, batch_size, sampler=None, shuffle=False, seed=42):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        pin_memory=(DEVICE == "cuda"),
    )


def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def log_reproducibility_info(results_dir):
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    gpu_memory_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else np.nan
    )
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "sklearn": sklearn.__version__,
        "optuna": optuna.__version__,
        "opencv": cv2.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": DEVICE,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "run_seed": RUN_SEED,
        "split_seed": SPLIT_SEED,
        "run_seeds": RUN_SEEDS,
        "use_amp": USE_AMP,
        "num_workers": NUM_WORKERS,
        "hpo_n_trials": HPO_N_TRIALS,
        "reoptimize_per_seed": REOPTIMIZE_PER_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "alpha": ALPHA,
        "pretrained": PRETRAINED,
        "xai_hpo_enable": XAI_HPO_ENABLE,
        "xai_hpo_n_samples": XAI_HPO_N_SAMPLES,
        "xai_hpo_ig_steps": XAI_HPO_IG_STEPS,
        "xai_hpo_weight": XAI_HPO_WEIGHT,
    }
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "reproducibility_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    pd.DataFrame([info]).to_csv(os.path.join(results_dir, "environment_summary.csv"), index=False)


set_seed(RUN_SEED)
log_reproducibility_info(RESULTS_DIR)

# DATA LOADING AND DATASET
if not os.path.exists(DATASET_DIR):
    print(f"WARNING: DATASET_DIR not found: {DATASET_DIR}")
    if os.path.exists("/kaggle/input"):
        print("Contents of /kaggle/input:")
        print(os.listdir("/kaggle/input"))
    else:
        print("/kaggle/input not found.")


def get_image_paths_and_labels(dataset_dir):
    image_paths = []
    labels = []
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if "_mask" in fname.lower():
                continue
            if fname.lower().endswith(valid_exts):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(CLASS_TO_IDX[class_name])

    return image_paths, labels


def get_mask_path(img_path):
    if not img_path:
        return None
    directory = os.path.dirname(img_path)
    filename = os.path.basename(img_path)
    base, ext = os.path.splitext(filename)

    exact_mask = os.path.join(directory, f"{base}_mask{ext}")
    if os.path.exists(exact_mask):
        return exact_mask

    if os.path.exists(directory):
        for fname in os.listdir(directory):
            fname_lower = fname.lower()
            if fname_lower.startswith(base.lower()) and "_mask" in fname_lower:
                return os.path.join(directory, fname)
    return None


def get_all_mask_paths(img_path):
    """Return all masks for an image: {base}_mask{ext}, {base}_mask_1{ext}, ...
    (multiple lesions). Empty list if none exist."""
    if not img_path:
        return []
    directory = os.path.dirname(img_path)
    base, _ = os.path.splitext(os.path.basename(img_path))

    masks = []
    if os.path.exists(directory):
        for fname in sorted(os.listdir(directory)):
            fname_lower = fname.lower()
            if fname_lower.startswith(base.lower()) and "_mask" in fname_lower:
                masks.append(os.path.join(directory, fname))
    return masks


def load_combined_mask(img_path, img_size=IMG_SIZE):
    """Load and merge (logical OR) all masks for an image, including multiple
    lesions. Returns a binary float32 array (img_size, img_size), or None if
    no mask exists."""
    paths = get_all_mask_paths(img_path)
    if not paths:
        return None

    combined = None
    for p in paths:
        arr = np.array(
            Image.open(p).convert("L").resize((img_size, img_size), resample=Image.NEAREST),
            dtype=np.float32,
        )
        arr = (arr > 0).astype(np.float32)
        combined = arr if combined is None else np.maximum(combined, arr)

    return combined


image_paths, labels = get_image_paths_and_labels(DATASET_DIR)
if len(image_paths) == 0:
    raise RuntimeError(f"No images found in DATASET_DIR={DATASET_DIR}")

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.30,
    stratify=labels,
    random_state=SPLIT_SEED,
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=SPLIT_SEED,
)

class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)
sample_weights = [class_weights_np[label] for label in train_labels]


def build_train_sampler(seed):
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=make_generator(seed),
    )


train_sampler = build_train_sampler(RUN_SEED)
print("Reproducible WeightedRandomSampler enabled.")

train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class BUSIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, is_training=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        # Blur scanner artifact in the bottom-right corner.
        w, h = image.size
        img_np = np.array(image)
        r_start_art, c_start_art = int(h * 0.80), int(w * 0.80)
        roi_art = img_np[r_start_art:, c_start_art:]
        if roi_art.size > 0:
            roi_blurred = cv2.GaussianBlur(roi_art, (51, 51), 0)
            img_np[r_start_art:, c_start_art:] = roi_blurred
        image = Image.fromarray(img_np)

        # Lesion-centric crop augmentation, training only.
        if self.is_training and label in [0, 1] and random.random() < 0.5:
            mask_path = get_mask_path(img_path)
            if mask_path:
                mask_np = np.array(Image.open(mask_path).convert("L"))
                coords = np.argwhere(mask_np > 0)
                if len(coords) > 0:
                    y0, x0 = coords.min(axis=0)
                    y1, x1 = coords.max(axis=0)
                    h_m, w_m = y1 - y0, x1 - x0
                    y0 = max(0, y0 - random.randint(0, max(1, int(h_m * 0.3))))
                    y1 = min(h, y1 + random.randint(0, max(1, int(h_m * 0.3))))
                    x0 = max(0, x0 - random.randint(0, max(1, int(w_m * 0.3))))
                    x1 = min(w, x1 + random.randint(0, max(1, int(w_m * 0.3))))
                    if x1 > x0 and y1 > y0:
                        image = image.crop((x0, y0, x1, y1))

        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = BUSIDataset(train_paths, train_labels, transform=train_transform, is_training=True)
val_dataset = BUSIDataset(val_paths, val_labels, transform=eval_transform, is_training=False)
test_dataset = BUSIDataset(test_paths, test_labels, transform=eval_transform, is_training=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# GENERAL UTILITIES
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improvement = score > self.best_score + self.min_delta
        else:
            improvement = score < self.best_score - self.min_delta

        if improvement:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def model_run_dir(results_dir, model_name, seed):
    safe_model = model_name.replace("/", "_")
    return os.path.join(results_dir, safe_model, f"seed_{seed}")


def create_timm_model(model_name, num_classes=3, drop_rate=0.0, pretrained=PRETRAINED):
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )


def _safe_auc(y_true_binary, y_score):
    y_true_binary = np.asarray(y_true_binary)
    if len(np.unique(y_true_binary)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_true_binary, y_score)
    except ValueError:
        return np.nan


def _safe_auprc(y_true_binary, y_score):
    y_true_binary = np.asarray(y_true_binary)
    if len(np.unique(y_true_binary)) < 2:
        return np.nan
    try:
        return average_precision_score(y_true_binary, y_score)
    except ValueError:
        return np.nan


def compute_clinical_metrics(y_true, y_pred, y_prob, class_names):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = None if y_prob is None else np.asarray(y_prob)
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    if y_prob is not None:
        try:
            metrics["auc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            metrics["auc_weighted"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        except ValueError:
            metrics["auc_macro"] = np.nan
            metrics["auc_weighted"] = np.nan
        try:
            metrics["auprc_macro"] = average_precision_score(y_true_bin, y_prob, average="macro")
            metrics["auprc_weighted"] = average_precision_score(y_true_bin, y_prob, average="weighted")
        except ValueError:
            metrics["auprc_macro"] = np.nan
            metrics["auprc_weighted"] = np.nan

    for i, cls in enumerate(class_names):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        tn = int(((y_pred != i) & (y_true != i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        metrics[f"{cls}_sensitivity"] = tp / (tp + fn + 1e-8)
        metrics[f"{cls}_specificity"] = tn / (tn + fp + 1e-8)
        metrics[f"{cls}_PPV"] = tp / (tp + fp + 1e-8)
        metrics[f"{cls}_NPV"] = tn / (tn + fn + 1e-8)
        if y_prob is not None:
            metrics[f"{cls}_AUC"] = _safe_auc(y_true_bin[:, i], y_prob[:, i])
            metrics[f"{cls}_AUPRC"] = _safe_auprc(y_true_bin[:, i], y_prob[:, i])

    return metrics


# XAI: INTEGRATED GRADIENTS AND POINTING GAME
def _get_grayscale_attention(model, input_tensor, label_idx, target_idx=None, steps=50):
    """Compute an Integrated Gradients saliency map.

    target_idx: class used for the IG backward pass (defaults to label_idx;
    XAI evaluation uses the model's predicted class instead).
    steps: integration steps (50 for final evaluation/plots, fewer for HPO
    via XAI_HPO_IG_STEPS).
    """
    baseline = torch.zeros_like(input_tensor)
    ig = torch.zeros_like(input_tensor[0])

    if target_idx is None:
        target_idx = label_idx

    model.eval()
    with torch.enable_grad():
        for i in range(1, steps + 1):
            interp = (baseline + (i / steps) * (input_tensor - baseline)).to(DEVICE)
            interp = interp.detach().requires_grad_(True)
            out = model(interp)
            model.zero_grad(set_to_none=True)
            out[0, target_idx].backward()
            ig += interp.grad.data[0]

    ig /= steps
    attr = (ig * (input_tensor[0] - baseline[0])).abs()
    sal, _ = torch.max(attr, dim=0)
    sal = sal.detach().cpu().numpy()
    sal = cv2.GaussianBlur(sal, (5, 5), 0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal, "Integrated Gradients"


def _compute_pointing_game_hit(attention_map, mask_np):
    if attention_map.shape != mask_np.shape:
        attention_map = cv2.resize(attention_map, (mask_np.shape[1], mask_np.shape[0]))

    mask_bin = mask_np > 0.5
    attr_bin = attention_map >= XAI_ATTRIBUTION_THRESHOLD

    peak_idx = np.unravel_index(np.argmax(attention_map), attention_map.shape)
    r, c = peak_idx
    r_start, r_end = max(0, r - 5), min(mask_np.shape[0], r + 6)
    c_start, c_end = max(0, c - 5), min(mask_np.shape[1], c + 6)
    region = mask_bin[r_start:r_end, c_start:c_end]
    hit = 1 if np.any(region) else 0

    energy_inside = (attention_map * mask_np).sum()
    total_energy = attention_map.sum() + 1e-8
    energy_fraction = energy_inside / total_energy

    intersection = np.logical_and(attr_bin, mask_bin).sum()
    dice = (2.0 * intersection) / (attr_bin.sum() + mask_bin.sum() + 1e-8)
    return hit, energy_fraction, dice


def compute_pointing_metrics_from_records(records_df, class_names):
    if records_df is None or len(records_df) == 0:
        results = {
            "pointing_game_acc": np.nan,
            "trustworthy_pg_acc": np.nan,
            "mean_energy_fraction": np.nan,
            "mean_xai_dice": np.nan,
        }
        for class_name in class_names:
            results[f"pg_acc_{class_name}"] = np.nan
            results[f"trust_acc_{class_name}"] = np.nan
            results[f"mean_energy_fraction_{class_name}"] = np.nan
            results[f"mean_xai_dice_{class_name}"] = np.nan
        return results

    results = {
        "pointing_game_acc": records_df["hit"].mean(),
        "trustworthy_pg_acc": records_df["trustworthy_hit"].mean(),
        "mean_energy_fraction": records_df["energy_fraction"].mean(),
        "mean_xai_dice": records_df["xai_dice"].mean(),
    }
    for i, class_name in enumerate(class_names):
        sub = records_df[records_df["y_true"] == i]
        results[f"pg_acc_{class_name}"] = sub["hit"].mean() if len(sub) else np.nan
        results[f"trust_acc_{class_name}"] = sub["trustworthy_hit"].mean() if len(sub) else np.nan
        results[f"mean_energy_fraction_{class_name}"] = sub["energy_fraction"].mean() if len(sub) else np.nan
        results[f"mean_xai_dice_{class_name}"] = sub["xai_dice"].mean() if len(sub) else np.nan
    return results


def evaluate_all_heatmaps_pointing_game(model, model_name, dataset, device, limit=None, return_records=False):
    model.eval()
    records = []
    processed_count = 0

    for idx in range(len(dataset)):
        img_tensor, label_idx = dataset[idx]
        mask_np = load_combined_mask(dataset.image_paths[idx], IMG_SIZE)
        if mask_np is None:
            continue

        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            pred_idx = torch.argmax(output, 1).item()
            is_correct = pred_idx == label_idx

        attention, _ = _get_grayscale_attention(
            model, img_tensor.unsqueeze(0).to(device), label_idx, target_idx=pred_idx
        )

        if mask_np.sum() < 1:
            continue

        hit, energy_frac, dice = _compute_pointing_game_hit(attention, mask_np)
        trustworthy_hit = 1 if (hit == 1 and is_correct) else 0
        records.append(
            {
                "idx": idx,
                "image_path": dataset.image_paths[idx],
                "y_true": int(label_idx),
                "y_pred": int(pred_idx),
                "hit": int(hit),
                "trustworthy_hit": int(trustworthy_hit),
                "energy_fraction": float(energy_frac),
                "xai_dice": float(dice),
            }
        )
        processed_count += 1

        if limit and processed_count >= limit:
            break

    records_df = pd.DataFrame(records)
    results = compute_pointing_metrics_from_records(records_df, CLASS_NAMES)
    if return_records:
        return results, records_df
    return results


def plot_model_heatmaps(model, model_name, dataset, device, results_dir=None):
    model.eval()
    class_samples = {i: None for i in range(len(CLASS_NAMES))}
    class_img_paths = {i: None for i in range(len(CLASS_NAMES))}
    class_best_energy = {i: -1.0 for i in range(len(CLASS_NAMES))}

    print(f"Searching XAI examples for {model_name}...")
    for idx in range(len(dataset)):
        img_path = dataset.image_paths[idx]
        label = dataset.labels[idx]
        mask_np_tmp = load_combined_mask(img_path, IMG_SIZE)

        if mask_np_tmp is not None:
            img_tensor, _ = dataset[idx]
            with torch.no_grad():
                out_tmp = model(img_tensor.unsqueeze(0).to(device))
                pred_idx_tmp = torch.argmax(out_tmp, 1).item()
            attention, _ = _get_grayscale_attention(
                model, img_tensor.unsqueeze(0).to(device), label, target_idx=pred_idx_tmp
            )
            if mask_np_tmp.sum() > 0:
                hit, energy_frac, _ = _compute_pointing_game_hit(attention, mask_np_tmp)
                if hit == 1 and energy_frac > class_best_energy[label]:
                    class_best_energy[label] = energy_frac
                    class_samples[label] = img_tensor
                    class_img_paths[label] = img_path

        if class_samples[label] is None:
            img_tensor, _ = dataset[idx]
            class_samples[label] = img_tensor
            class_img_paths[label] = img_path

    fig, axes = plt.subplots(len(CLASS_NAMES), 4, figsize=(20, 15))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    pointing_game_results = {}

    for row, (label_idx, img_tensor) in enumerate(class_samples.items()):
        if img_tensor is None:
            for ax in axes[row]:
                ax.set_visible(False)
            continue

        img_np = np.clip(std * img_tensor.permute(1, 2, 0).numpy() + mean, 0, 1)

        with torch.no_grad():
            out_row = model(img_tensor.unsqueeze(0).to(device))
            pred_idx_row = torch.argmax(out_row, 1).item()

        attention, method = _get_grayscale_attention(
            model, img_tensor.unsqueeze(0).to(device), label_idx, target_idx=pred_idx_row
        )
        mask_np = load_combined_mask(class_img_paths[label_idx], IMG_SIZE)
        if mask_np is not None:
            if mask_np.sum() < 1:
                hit, energy_frac = None, None
            else:
                hit, energy_frac, dice = _compute_pointing_game_hit(attention, mask_np)
            pointing_game_results[IDX_TO_CLASS[label_idx]] = (
                f"HIT ({energy_frac:.1%}, Dice={dice:.3f})"
                if hit == 1
                else (f"MISS ({energy_frac:.1%}, Dice={dice:.3f})" if hit == 0 else "N/A")
            )
        else:
            mask_np, hit = np.zeros((IMG_SIZE, IMG_SIZE)), None

        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)[..., ::-1]) / 255
        vis = np.clip(heatmap * 0.5 + img_np * 0.5, 0, 1)
        peak_idx = np.unravel_index(np.argmax(attention), attention.shape)
        overlay = np.ascontiguousarray((vis * 255).astype(np.uint8))
        color = (0, 255, 0) if hit == 1 else ((255, 0, 0) if hit == 0 else (128, 128, 128))
        cv2.drawMarker(overlay, (peak_idx[1], peak_idx[0]), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=3)

        img_name = os.path.basename(class_img_paths[label_idx])
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_ylabel(f"CLASS: {IDX_TO_CLASS[label_idx]}\n{img_name}", fontsize=10, fontweight="bold")
        axes[row, 1].imshow(mask_np, cmap="gray")
        axes[row, 1].scatter(peak_idx[1], peak_idx[0], color="yellow", marker="x", s=100)
        axes[row, 1].set_title("Ground Truth Mask")
        axes[row, 2].imshow(vis)
        axes[row, 2].set_title(f"XAI: {method}")
        axes[row, 3].imshow(overlay / 255.0)
        axes[row, 3].set_title(f"Pointing: {pointing_game_results.get(IDX_TO_CLASS[label_idx], 'N/A')}")
        for i in range(4):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    plt.tight_layout()
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, "pointing_game_examples.png"), dpi=200)
    plt.close()


def save_all_test_heatmaps(model, model_name, dataset, device, results_dir, limit=None):
    model.eval()
    save_path = os.path.join(results_dir, "all_heatmaps")
    os.makedirs(save_path, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for idx in range(len(dataset)):
        if limit and idx >= limit:
            break
        img_tensor, label_idx = dataset[idx]
        img_path = dataset.image_paths[idx]
        img_name = os.path.basename(img_path)

        with torch.no_grad():
            out = model(img_tensor.unsqueeze(0).to(device))
            pred_idx = torch.argmax(out, 1).item()

        attention, method = _get_grayscale_attention(
            model, img_tensor.unsqueeze(0).to(device), label_idx, target_idx=pred_idx
        )
        img_np = np.clip(std * img_tensor.permute(1, 2, 0).numpy() + mean, 0, 1)

        mask_np = load_combined_mask(img_path, IMG_SIZE)
        if mask_np is not None:
            hit, energy_frac, dice = _compute_pointing_game_hit(attention, mask_np)
            status = f"HIT_{energy_frac:.1%}_Dice_{dice:.3f}" if hit == 1 else f"MISS_{energy_frac:.1%}_Dice_{dice:.3f}"
        else:
            mask_np = np.zeros((IMG_SIZE, IMG_SIZE))
            status = "N_A"

        heatmap = np.float32(cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)[..., ::-1]) / 255
        vis = np.clip(heatmap * 0.5 + img_np * 0.5, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original: {IDX_TO_CLASS[label_idx]}")
        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[2].imshow(vis)
        axes[2].set_title(f"{method} ({status})")
        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"xai_{idx:03d}_{img_name}"), dpi=200)
        plt.close()


# TRAINING AND EVALUATION
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = 0

    for images, labels_batch in loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        all_preds.extend(torch.argmax(outputs, 1).detach().cpu().numpy())
        all_labels.extend(labels_batch.detach().cpu().numpy())

    return running_loss / total_samples, accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels_batch in loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(outputs, 1).detach().cpu().numpy())
        all_labels.extend(labels_batch.detach().cpu().numpy())

    return (
        running_loss / len(loader.dataset),
        accuracy_score(all_labels, all_preds),
        f1_score(all_labels, all_preds, average="macro", zero_division=0),
    )


@torch.no_grad()
def evaluate_full(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels_batch in loader:
        images = images.to(device)
        labels_batch = labels_batch.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        probs = torch.softmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(outputs, 1).detach().cpu().numpy())
        all_labels.extend(labels_batch.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    all_probs = np.asarray(all_probs)
    y_true = np.asarray(all_labels)

    try:
        auc_macro = roc_auc_score(y_true, all_probs, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y_true, all_probs, multi_class="ovr", average="weighted")
    except ValueError:
        auc_macro = np.nan
        auc_weighted = np.nan

    return (
        running_loss / len(loader.dataset),
        accuracy_score(all_labels, all_preds),
        all_labels,
        all_preds,
        all_probs,
        auc_macro,
        auc_weighted,
    )


def select_pointing_game_hpo_samples(dataset, max_samples=XAI_HPO_N_SAMPLES):
    """Select a small, fixed set of masked benign/malignant images, once, to
    estimate the pointing game quickly inside the Optuna objective."""
    samples = []
    if max_samples <= 0:
        return samples

    for idx in range(len(dataset)):
        label_idx = dataset.labels[idx]
        if label_idx not in (0, 1):  # benign, malignant: classes with a lesion mask
            continue

        mask_np = load_combined_mask(dataset.image_paths[idx], IMG_SIZE)
        if mask_np is None or mask_np.sum() < 1:
            continue

        img_tensor, _ = dataset[idx]
        samples.append((img_tensor, mask_np, label_idx))
        if len(samples) >= max_samples:
            break

    return samples


# Computed once, outside objective(), to avoid repeating mask lookup/loading
# on every Optuna trial.
HPO_XAI_SAMPLES = (
    select_pointing_game_hpo_samples(val_dataset, XAI_HPO_N_SAMPLES) if XAI_HPO_ENABLE else []
)
if XAI_HPO_ENABLE and len(HPO_XAI_SAMPLES) == 0:
    print("WARNING: XAI_HPO_ENABLE=True but no masked images found in val_dataset; "
          "the XAI term of the Optuna objective will be disabled.")


def compute_hpo_pointing_score(model, samples, device, steps=XAI_HPO_IG_STEPS):
    """Average Pointing Game accuracy over a small fixed reference set, using
    Integrated Gradients with few steps to keep the per-trial cost low."""
    if not samples:
        return 0.0

    was_training = model.training
    model.eval()
    hits = []

    for img_tensor, mask_np, label_idx in samples:
        input_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred_idx = torch.argmax(output, 1).item()

        attention, _ = _get_grayscale_attention(
            model, input_tensor, label_idx, target_idx=pred_idx, steps=steps
        )
        hit, _, _ = _compute_pointing_game_hit(attention, mask_np)
        hits.append(hit)

    if was_training:
        model.train()

    return float(np.mean(hits)) if hits else 0.0


def objective(trial, model_name, run_seed=RUN_SEED):
    trial_seed = run_seed + trial.number * 1000
    set_seed(trial_seed)

    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    loader_train = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        sampler=build_train_sampler(trial_seed),
        seed=trial_seed,
    )
    loader_val = make_dataloader(val_dataset, batch_size=batch_size, shuffle=False, seed=trial_seed)

    model = create_timm_model(model_name, pretrained=PRETRAINED, num_classes=3, drop_rate=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_score = -np.inf
    for epoch in range(8):
        train_one_epoch(model, loader_train, criterion, optimizer, DEVICE)
        _, val_acc, val_f1 = evaluate(model, loader_val, criterion, DEVICE)

        classification_score = 0.7 * val_f1 + 0.3 * val_acc

        # Optional XAI (pointing game) term: rewards models whose Integrated
        # Gradients attribution falls inside the lesion, on a small fixed set.
        if XAI_HPO_ENABLE and HPO_XAI_SAMPLES:
            pg_score = compute_hpo_pointing_score(model, HPO_XAI_SAMPLES, DEVICE, steps=XAI_HPO_IG_STEPS)
            score = (1 - XAI_HPO_WEIGHT) * classification_score + XAI_HPO_WEIGHT * pg_score
        else:
            score = classification_score

        best_score = max(best_score, score)

        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_score


def optimize_model(model_name, run_seed=RUN_SEED):
    sampler = optuna.samplers.TPESampler(seed=run_seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )
    study.optimize(lambda t: objective(t, model_name, run_seed), n_trials=HPO_N_TRIALS)
    return study.best_params


def best_params_path(model_name, seed=None):
    params_dir = os.path.join(RESULTS_DIR, "best_params")
    os.makedirs(params_dir, exist_ok=True)
    safe_model = model_name.replace("/", "_")
    if REOPTIMIZE_PER_SEED:
        return os.path.join(params_dir, f"{safe_model}_seed_{seed}.json")
    return os.path.join(params_dir, f"{safe_model}.json")


def load_or_optimize_params(model_name, seed):
    path = best_params_path(model_name, seed)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    params = optimize_model(model_name, run_seed=seed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return params


def train_final_model(model_name, best_params, run_seed=RUN_SEED):
    set_seed(run_seed)

    batch_size = best_params["batch_size"]
    lr = best_params["lr"]
    weight_decay = best_params["weight_decay"]
    dropout = best_params["dropout"]

    loader_train = make_dataloader(
        train_dataset,
        batch_size=batch_size,
        sampler=build_train_sampler(run_seed),
        seed=run_seed,
    )
    loader_val = make_dataloader(val_dataset, batch_size=batch_size, shuffle=False, seed=run_seed)
    loader_test = make_dataloader(test_dataset, batch_size=batch_size, shuffle=False, seed=run_seed)

    model = create_timm_model(model_name, pretrained=PRETRAINED, num_classes=3, drop_rate=dropout).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    es = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, mode="max")

    run_dir = model_run_dir(RESULTS_DIR, model_name, run_seed)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, f"best_{model_name.replace('/', '_')}_seed_{run_seed}.pth")

    best_score = -np.inf
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "score": [],
        "lr": [],
    }
    scaler = torch.amp.GradScaler("cuda") if USE_AMP else None
    training_start_time = time.perf_counter()

    for epoch in range(EPOCHS):
        t_loss, t_acc = train_one_epoch(model, loader_train, criterion, optimizer, DEVICE, scaler=scaler)
        v_loss, v_acc, v_f1 = evaluate(model, loader_val, criterion, DEVICE)

        # Same criterion as Optuna, for selecting the final checkpoint.
        score = 0.7 * v_f1 + 0.3 * v_acc

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)
        history["score"].append(score)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"[{model_name} | seed={run_seed}] Ep {epoch + 1}/{EPOCHS} | "
            f"Val F1: {v_f1:.4f} | Val Acc: {v_acc:.4f} | Score: {score:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)

        scheduler.step(score)
        if es.step(score):
            break

    training_time_sec = time.perf_counter() - training_start_time
    model.load_state_dict(safe_torch_load(best_path, map_location=DEVICE))
    test_loss, test_acc, y_true, y_pred, y_prob, auc_m, auc_w = evaluate_full(model, loader_test, criterion, DEVICE)

    return {
        "model_name": model_name,
        "seed": run_seed,
        "best_params": best_params,
        "checkpoint_path": best_path,
        "training_time_sec": training_time_sec,
        "epochs_trained": len(history["epoch"]),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "test_auc_macro": auc_m,
        "test_auc_weighted": auc_w,
        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred),
        "history": pd.DataFrame(history),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# BOOTSTRAP / 95% CI
def bootstrap_classification_metric_ci(
    y_true,
    y_pred,
    y_prob,
    class_names,
    n_boot=1000,
    seed=2026,
    alpha=0.05,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = None if y_prob is None else np.asarray(y_prob)
    rng = np.random.default_rng(seed)

    base_metrics = compute_clinical_metrics(y_true, y_pred, y_prob, class_names)
    boot_values = {k: [] for k in base_metrics.keys()}
    idx_by_class = [np.where(y_true == c)[0] for c in range(len(class_names))]

    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(idxs, size=len(idxs), replace=True)
                for idxs in idx_by_class
                if len(idxs) > 0
            ]
        )
        rng.shuffle(idx)
        probs_b = None if y_prob is None else y_prob[idx]
        metrics_b = compute_clinical_metrics(y_true[idx], y_pred[idx], probs_b, class_names)

        for key, value in metrics_b.items():
            if np.isfinite(value):
                boot_values[key].append(value)

    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)
    rows = []
    for metric, value in base_metrics.items():
        vals = np.asarray(boot_values.get(metric, []), dtype=float)
        if len(vals) == 0:
            ci_low, ci_high = np.nan, np.nan
        else:
            ci_low, ci_high = np.percentile(vals, [lo_q, hi_q])
        rows.append(
            {
                "metric": metric,
                "value": value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_method": f"stratified bootstrap, n={n_boot}",
            }
        )

    return pd.DataFrame(rows)


def bootstrap_pointing_metric_ci(records_df, class_names, n_boot=1000, seed=2026, alpha=0.05):
    if records_df is None or len(records_df) == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    records_df = records_df.reset_index(drop=True)
    base_metrics = compute_pointing_metrics_from_records(records_df, class_names)
    boot_values = {k: [] for k in base_metrics.keys()}
    idx_by_class = [records_df.index[records_df["y_true"] == c].to_numpy() for c in range(len(class_names))]

    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(idxs, size=len(idxs), replace=True)
                for idxs in idx_by_class
                if len(idxs) > 0
            ]
        )
        rng.shuffle(idx)
        metrics_b = compute_pointing_metrics_from_records(records_df.iloc[idx], class_names)
        for key, value in metrics_b.items():
            if np.isfinite(value):
                boot_values[key].append(value)

    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)
    rows = []
    for metric, value in base_metrics.items():
        vals = np.asarray(boot_values.get(metric, []), dtype=float)
        if len(vals) == 0:
            ci_low, ci_high = np.nan, np.nan
        else:
            ci_low, ci_high = np.percentile(vals, [lo_q, hi_q])
        rows.append(
            {
                "metric": metric,
                "value": value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_method": f"stratified bootstrap, n={n_boot}",
            }
        )

    return pd.DataFrame(rows)


# PAIRED TESTS: MCNEMAR, DELONG, HOLM-BONFERRONI
def compute_midrank(x):
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(predictions_sorted_transposed, label_1_count):
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both positive and negative samples.")

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for row in range(k):
        tx[row, :] = compute_midrank(positive_examples[row, :])
        ty[row, :] = compute_midrank(negative_examples[row, :])
        tz[row, :] = compute_midrank(predictions_sorted_transposed[row, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true_binary, pred_a, pred_b):
    y_true_binary = np.asarray(y_true_binary).astype(int)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    if len(np.unique(y_true_binary)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    order = np.argsort(-y_true_binary)
    preds = np.vstack((pred_a, pred_b))[:, order]
    label_1_count = int(y_true_binary.sum())
    aucs, cov = fast_delong(preds, label_1_count)

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        p_value = 1.0 if np.isclose(diff, 0) else np.nan
    else:
        z = abs(diff) / np.sqrt(var)
        p_value = 2 * (1 - norm.cdf(z))

    return aucs[0], aucs[1], diff, p_value


def mcnemar_exact_test(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only

    p_value = 1.0 if discordant == 0 else binomtest(min(a_only, b_only), discordant, p=0.5).pvalue
    statistic = 0.0 if discordant == 0 else (abs(a_only - b_only) - 1) ** 2 / discordant
    return statistic, p_value, a_only, b_only, discordant


def apply_holm_bonferroni(df, p_col="p_value", family_col="test_family", alpha=0.05):
    out = df.copy()
    out["p_holm"] = np.nan
    out["reject_holm"] = False

    for family, group in out.groupby(family_col):
        idx = group.index[group[p_col].notna()].to_numpy()
        if len(idx) == 0:
            continue

        pvals = out.loc[idx, p_col].astype(float).to_numpy()
        order = np.argsort(pvals)
        adjusted = np.empty(len(pvals), dtype=float)
        running_max = 0.0

        for rank, pos in enumerate(order):
            adj = (len(pvals) - rank) * pvals[pos]
            running_max = max(running_max, adj)
            adjusted[pos] = min(running_max, 1.0)

        out.loc[idx, "p_holm"] = adjusted
        out.loc[idx, "reject_holm"] = adjusted < alpha

    return out


# SAVING, PLOTS AND AGGREGATION
def save_prediction_artifacts(res, model_name, seed, results_dir):
    run_dir = model_run_dir(results_dir, model_name, seed)
    os.makedirs(run_dir, exist_ok=True)

    y_true = np.asarray(res["y_true"])
    y_pred = np.asarray(res["y_pred"])
    y_prob = np.asarray(res["y_prob"])

    pred_df = pd.DataFrame(
        {
            "image_path": test_paths if len(test_paths) == len(y_true) else np.arange(len(y_true)),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    for i, cls in enumerate(CLASS_NAMES):
        pred_df[f"prob_{cls}"] = y_prob[:, i]

    pred_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)
    res["history"].to_csv(os.path.join(run_dir, "history.csv"), index=False)

    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(res["report"])

    with open(os.path.join(run_dir, "best_params_used.json"), "w", encoding="utf-8") as f:
        json.dump(res["best_params"], f, indent=2)


def save_metrics_artifacts(metrics, metrics_ci_df, model_name, seed, results_dir):
    run_dir = model_run_dir(results_dir, model_name, seed)
    os.makedirs(run_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)
    if metrics_ci_df is not None and len(metrics_ci_df) > 0:
        metrics_ci_df.to_csv(os.path.join(run_dir, "metrics_with_ci.csv"), index=False)


def plot_training_results(history_df, model_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    epochs = history_df["epoch"]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_df["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history_df["val_loss"], label="Val Loss", marker="s")
    plt.title(f"{model_name} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_df["train_acc"], label="Train Acc", marker="o")
    plt.plot(epochs, history_df["val_acc"], label="Val Acc", marker="s")
    plt.title(f"{model_name} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "training_curves.png"), dpi=200)
    plt.close()


def plot_confusion_matrix_heatmap(cm, class_names, model_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=200)
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, model_name, results_dir, filename_prefix="roc_curves"):
    os.makedirs(results_dir, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = len(class_names)

    if len(y_true) == 0 or y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        print(f"Could not plot ROC curves for {model_name}: invalid shapes.")
        return pd.DataFrame()

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_true_bin.shape[1] == 1 and n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin[:, 0], y_true_bin[:, 0]])

    roc_rows = []
    fpr_by_class = {}
    tpr_by_class = {}
    auc_by_class = {}

    plt.figure(figsize=(8, 7))

    for class_idx, class_name in enumerate(class_names):
        y_binary = y_true_bin[:, class_idx]
        if len(np.unique(y_binary)) < 2:
            print(f"ROC skipped for {model_name} / {class_name}: missing positive or negative class.")
            continue

        fpr, tpr, thresholds = roc_curve(y_binary, y_prob[:, class_idx])
        class_auc = auc(fpr, tpr)
        fpr_by_class[class_idx] = fpr
        tpr_by_class[class_idx] = tpr
        auc_by_class[class_idx] = class_auc

        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={class_auc:.3f})")
        for point_idx, (fpr_value, tpr_value, threshold_value) in enumerate(zip(fpr, tpr, thresholds)):
            roc_rows.append(
                {
                    "curve": class_name,
                    "class_idx": class_idx,
                    "point_idx": point_idx,
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "threshold": float(threshold_value),
                    "auc": float(class_auc),
                }
            )

    if len(roc_rows) == 0:
        plt.close()
        print(f"No ROC generated for {model_name}: no class had both positives and negatives.")
        return pd.DataFrame()

    try:
        fpr_micro, tpr_micro, thresholds_micro = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, linestyle=":", lw=2.5, label=f"micro-average (AUC={auc_micro:.3f})")
        for point_idx, (fpr_value, tpr_value, threshold_value) in enumerate(
            zip(fpr_micro, tpr_micro, thresholds_micro)
        ):
            roc_rows.append(
                {
                    "curve": "micro_average",
                    "class_idx": -1,
                    "point_idx": point_idx,
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "threshold": float(threshold_value),
                    "auc": float(auc_micro),
                }
            )
    except ValueError as exc:
        print(f"Could not compute micro-average ROC for {model_name}: {exc}")

    if fpr_by_class:
        all_fpr = np.unique(np.concatenate(list(fpr_by_class.values())))
        mean_tpr = np.zeros_like(all_fpr)
        for class_idx in fpr_by_class:
            mean_tpr += np.interp(all_fpr, fpr_by_class[class_idx], tpr_by_class[class_idx])
        mean_tpr /= len(fpr_by_class)
        auc_macro = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, linestyle="--", lw=2.5, label=f"macro-average (AUC={auc_macro:.3f})")
        for point_idx, (fpr_value, tpr_value) in enumerate(zip(all_fpr, mean_tpr)):
            roc_rows.append(
                {
                    "curve": "macro_average",
                    "class_idx": -2,
                    "point_idx": point_idx,
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "threshold": np.nan,
                    "auc": float(auc_macro),
                }
            )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC one-vs-rest")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{filename_prefix}.png"), dpi=300)
    plt.close()

    roc_df = pd.DataFrame(roc_rows)
    roc_df.to_csv(os.path.join(results_dir, f"{filename_prefix}_points.csv"), index=False)
    return roc_df


def plot_seed_averaged_roc_curves(aggregated_predictions, class_names, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for model_name, pred in aggregated_predictions.items():
        model_dir = os.path.join(results_dir, "seed_averaged_roc", model_name)
        roc_df = plot_roc_curves(
            pred["y_true"],
            pred["y_prob"],
            class_names,
            f"{model_name} seed-averaged",
            model_dir,
            filename_prefix="roc_curves_seed_averaged",
        )
        if len(roc_df) > 0:
            roc_df.insert(0, "Model", model_name)
            roc_df.insert(1, "n_seeds_used", len(pred["used_seeds"]))
            rows.append(roc_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(results_dir, "roc_curves_seed_averaged_points.csv"), index=False)
    return out


def plot_precision_recall_curves(y_true, y_prob, class_names, model_name, results_dir, filename_prefix="pr_curves"):
    os.makedirs(results_dir, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = len(class_names)

    if len(y_true) == 0 or y_prob.ndim != 2 or y_prob.shape[1] != n_classes:
        print(f"Could not plot PR curves for {model_name}: invalid shapes.")
        return pd.DataFrame()

    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    if y_true_bin.shape[1] == 1 and n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin[:, 0], y_true_bin[:, 0]])

    rows = []
    plt.figure(figsize=(8, 7))

    for class_idx, class_name in enumerate(class_names):
        y_binary = y_true_bin[:, class_idx]
        if len(np.unique(y_binary)) < 2:
            print(f"PR skipped for {model_name} / {class_name}: missing positive or negative class.")
            continue

        precision, recall, thresholds = precision_recall_curve(y_binary, y_prob[:, class_idx])
        class_auprc = average_precision_score(y_binary, y_prob[:, class_idx])
        plt.plot(recall, precision, lw=2, label=f"{class_name} (AUPRC={class_auprc:.3f})")

        threshold_values = np.append(thresholds, np.nan)
        for point_idx, (recall_value, precision_value, threshold_value) in enumerate(
            zip(recall, precision, threshold_values)
        ):
            rows.append(
                {
                    "curve": class_name,
                    "class_idx": class_idx,
                    "point_idx": point_idx,
                    "recall": float(recall_value),
                    "precision": float(precision_value),
                    "threshold": float(threshold_value) if np.isfinite(threshold_value) else np.nan,
                    "auprc": float(class_auprc),
                }
            )

    if len(rows) == 0:
        plt.close()
        print(f"No PR generated for {model_name}: no class had both positives and negatives.")
        return pd.DataFrame()

    try:
        precision_micro, recall_micro, thresholds_micro = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
        auprc_micro = average_precision_score(y_true_bin, y_prob, average="micro")
        plt.plot(recall_micro, precision_micro, linestyle=":", lw=2.5, label=f"micro-average (AUPRC={auprc_micro:.3f})")
        threshold_values = np.append(thresholds_micro, np.nan)
        for point_idx, (recall_value, precision_value, threshold_value) in enumerate(
            zip(recall_micro, precision_micro, threshold_values)
        ):
            rows.append(
                {
                    "curve": "micro_average",
                    "class_idx": -1,
                    "point_idx": point_idx,
                    "recall": float(recall_value),
                    "precision": float(precision_value),
                    "threshold": float(threshold_value) if np.isfinite(threshold_value) else np.nan,
                    "auprc": float(auprc_micro),
                }
            )
    except ValueError as exc:
        print(f"Could not compute micro-average PR for {model_name}: {exc}")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall one-vs-rest")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{filename_prefix}.png"), dpi=300)
    plt.close()

    pr_df = pd.DataFrame(rows)
    pr_df.to_csv(os.path.join(results_dir, f"{filename_prefix}_points.csv"), index=False)
    return pr_df


def plot_seed_averaged_pr_curves(aggregated_predictions, class_names, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for model_name, pred in aggregated_predictions.items():
        model_dir = os.path.join(results_dir, "seed_averaged_pr", model_name)
        pr_df = plot_precision_recall_curves(
            pred["y_true"],
            pred["y_prob"],
            class_names,
            f"{model_name} seed-averaged",
            model_dir,
            filename_prefix="pr_curves_seed_averaged",
        )
        if len(pr_df) > 0:
            pr_df.insert(0, "Model", model_name)
            pr_df.insert(1, "n_seeds_used", len(pred["used_seeds"]))
            rows.append(pr_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(results_dir, "pr_curves_seed_averaged_points.csv"), index=False)
    return out


def generate_global_comparison(all_results, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    summary_data = []
    for model_name, res in all_results.items():
        summary_data.append(
            {
                "Model": model_name,
                "Test Acc": res["test_acc"],
                "Test F1 Macro": res["test_f1_macro"],
                "AUC Macro": res["test_auc_macro"],
                "AUPRC Macro": res.get("auprc_macro", np.nan),
                "Training Time Sec": res.get("training_time_sec", np.nan),
                "Epochs Trained": res.get("epochs_trained", np.nan),
                "Trustworthy PG": res.get("trustworthy_pg_acc", np.nan),
                "Mean Energy Fraction": res.get("mean_energy_fraction", np.nan),
                "Mean XAI Dice": res.get("mean_xai_dice", np.nan),
            }
        )

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(results_dir, "global_comparison.csv"), index=False)

    if len(df_summary) == 0:
        return

    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_summary))
    width = 0.25
    plt.bar(x - width, df_summary["Test Acc"], width, label="Accuracy")
    plt.bar(x, df_summary["Test F1 Macro"], width, label="F1 Macro")
    plt.bar(x + width, df_summary["AUC Macro"], width, label="AUC Macro")
    plt.title("Model Global Comparison")
    plt.xticks(x, df_summary["Model"])
    plt.ylabel("Score")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "global_comparison_plot.png"), dpi=200)
    plt.close()


def plot_overfitting_analysis(all_results, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    gap_data = []
    for model_name, res in all_results.items():
        hist = res["history"]
        gap_data.append(
            {
                "Model": model_name,
                "Acc Gap": hist["train_acc"].iloc[-1] - hist["val_acc"].iloc[-1],
                "Loss Gap": hist["val_loss"].iloc[-1] - hist["train_loss"].iloc[-1],
            }
        )

    df_gap = pd.DataFrame(gap_data)
    df_gap.to_csv(os.path.join(results_dir, "overfitting_gaps.csv"), index=False)
    if len(df_gap) == 0:
        return

    plt.figure(figsize=(10, 6))
    x = np.arange(len(df_gap))
    width = 0.35
    plt.bar(x - width / 2, df_gap["Acc Gap"], width, label="Acc Gap (Train-Val)", color="orange")
    plt.bar(x + width / 2, df_gap["Loss Gap"], width, label="Loss Gap (Val-Train)", color="red")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Overfitting Analysis - Metric Gaps")
    plt.xticks(x, df_gap["Model"])
    plt.ylabel("Gap Value")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "overfitting_analysis.png"), dpi=200)
    plt.close()


def read_metric_ci_file(model_name, seed):
    path = os.path.join(model_run_dir(RESULTS_DIR, model_name, seed), "metrics_with_ci.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    row = {"Model": model_name, "seed": seed}
    for _, r in df.iterrows():
        row[r["metric"]] = r["value"]
        row[f"{r['metric']}_ci_low"] = r["ci_low"]
        row[f"{r['metric']}_ci_high"] = r["ci_high"]
    return row


def summarize_seed_runs(models, seeds):
    rows = []
    for model_name in models:
        for seed in seeds:
            row = read_metric_ci_file(model_name, seed)
            if row is not None:
                rows.append(row)

    if not rows:
        print("No per-seed metrics with CI available to aggregate.")
        return pd.DataFrame(), pd.DataFrame()

    metrics_by_seed = pd.DataFrame(rows)
    metrics_by_seed.to_csv(os.path.join(RESULTS_DIR, "metrics_by_seed.csv"), index=False)

    metric_cols = [
        c
        for c in metrics_by_seed.columns
        if c not in ["Model", "seed"] and not c.endswith("_ci_low") and not c.endswith("_ci_high")
    ]

    summary_rows = []
    for model_name, group in metrics_by_seed.groupby("Model"):
        for metric in metric_cols:
            vals = pd.to_numeric(group[metric], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            mean = vals.mean()
            sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
            summary_rows.append(
                {
                    "Model": model_name,
                    "metric": metric,
                    "n_seeds": len(vals),
                    "mean": mean,
                    "sd": sd,
                    "mean_sd": f"{mean:.4f} +/- {sd:.4f}",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "metrics_mean_sd.csv"), index=False)
    return metrics_by_seed, summary_df


def load_seed_averaged_predictions(models, seeds):
    aggregated = {}
    for model_name in models:
        prob_list = []
        y_true_ref = None
        image_paths_ref = None
        used_seeds = []

        for seed in seeds:
            pred_path = os.path.join(model_run_dir(RESULTS_DIR, model_name, seed), "predictions.csv")
            if not os.path.exists(pred_path):
                print(f"No predictions found for {model_name}, seed={seed}: {pred_path}")
                continue

            df = pd.read_csv(pred_path)
            prob_cols = [f"prob_{cls}" for cls in CLASS_NAMES]
            probs = df[prob_cols].to_numpy(dtype=float)
            y_true = df["y_true"].to_numpy(dtype=int)

            if y_true_ref is None:
                y_true_ref = y_true
                image_paths_ref = df["image_path"].to_numpy()
            elif not np.array_equal(y_true_ref, y_true):
                raise ValueError(f"Test set changed for {model_name}, seed={seed}. Check SPLIT_SEED.")

            prob_list.append(probs)
            used_seeds.append(seed)

        if prob_list:
            y_prob = np.mean(np.stack(prob_list, axis=0), axis=0)
            y_pred = np.argmax(y_prob, axis=1)
            aggregated[model_name] = {
                "y_true": y_true_ref,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "image_path": image_paths_ref,
                "used_seeds": used_seeds,
            }

    return aggregated


def save_seed_averaged_prediction_cis(aggregated_predictions):
    rows = []
    for model_name, pred in aggregated_predictions.items():
        ci_df = bootstrap_classification_metric_ci(
            pred["y_true"],
            pred["y_pred"],
            pred["y_prob"],
            CLASS_NAMES,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + 1000,
            alpha=ALPHA,
        )
        ci_df.insert(0, "Model", model_name)
        ci_df.insert(1, "n_seeds_used", len(pred["used_seeds"]))
        rows.append(ci_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(RESULTS_DIR, "metrics_seed_averaged_bootstrap_ci.csv"), index=False)
    return out


def load_seed_averaged_pointing_records(models, seeds):
    aggregated = {}
    for model_name in models:
        dfs = []
        for seed in seeds:
            path = os.path.join(model_run_dir(RESULTS_DIR, model_name, seed), "pointing_game_records.csv")
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
            else:
                print(f"No pointing game records found for {model_name}, seed={seed}: {path}")
        if dfs:
            aggregated[model_name] = pd.concat(dfs, ignore_index=True)
    return aggregated


def save_seed_averaged_pointing_cis(pointing_records_by_model):
    rows = []
    for model_name, records_df in pointing_records_by_model.items():
        ci_df = bootstrap_pointing_metric_ci(
            records_df,
            CLASS_NAMES,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + 2000,
            alpha=ALPHA,
        )
        if len(ci_df) == 0:
            continue
        ci_df.insert(0, "Model", model_name)
        ci_df.insert(1, "n_seeds_used", records_df["idx"].notna().sum() and None)
        rows.append(ci_df)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    if "n_seeds_used" in out.columns and out["n_seeds_used"].isna().all():
        out = out.drop(columns=["n_seeds_used"])
    out.to_csv(os.path.join(RESULTS_DIR, "pointing_game_seed_averaged_bootstrap_ci.csv"), index=False)
    return out


def paired_model_tests(aggregated_predictions, class_names, alpha=0.05):
    rows = []
    model_names = list(aggregated_predictions.keys())

    if len(model_names) < 2:
        raise ValueError("At least two models' predictions are required.")

    for model_a, model_b in combinations(model_names, 2):
        a = aggregated_predictions[model_a]
        b = aggregated_predictions[model_b]
        y_true = np.asarray(a["y_true"])

        if not np.array_equal(y_true, np.asarray(b["y_true"])):
            raise ValueError(f"Different test sets between {model_a} and {model_b}.")

        stat, p_value, a_only, b_only, discordant = mcnemar_exact_test(y_true, a["y_pred"], b["y_pred"])
        rows.append(
            {
                "test_family": "mcnemar_accuracy",
                "comparison": f"{model_a} vs {model_b}",
                "class": "overall",
                "statistic": stat,
                "p_value": p_value,
                "effect": a_only - b_only,
                "detail": f"{model_a}_only_correct={a_only}; {model_b}_only_correct={b_only}; discordant={discordant}",
            }
        )

        for cls_idx, cls_name in enumerate(class_names):
            y_bin = (y_true == cls_idx).astype(int)
            auc_a, auc_b, diff, p_auc = delong_roc_test(
                y_bin,
                a["y_prob"][:, cls_idx],
                b["y_prob"][:, cls_idx],
            )
            rows.append(
                {
                    "test_family": "delong_auc_ovr",
                    "comparison": f"{model_a} vs {model_b}",
                    "class": cls_name,
                    "statistic": diff,
                    "p_value": p_auc,
                    "effect": diff,
                    "detail": f"AUC_{model_a}={auc_a}; AUC_{model_b}={auc_b}",
                }
            )

    tests_df = pd.DataFrame(rows)
    return apply_holm_bonferroni(tests_df, alpha=alpha)


def make_publication_table(summary_df, seedavg_ci_df):
    if summary_df.empty or seedavg_ci_df.empty:
        return pd.DataFrame()

    ci_small = seedavg_ci_df[["Model", "metric", "ci_low", "ci_high"]].copy()
    out = summary_df.merge(ci_small, on=["Model", "metric"], how="left")
    out["formatted"] = out.apply(
        lambda r: f"{r['mean']:.4f} +/- {r['sd']:.4f} (95% CI {r['ci_low']:.4f}-{r['ci_high']:.4f})"
        if pd.notna(r.get("ci_low", np.nan))
        else f"{r['mean']:.4f} +/- {r['sd']:.4f}",
        axis=1,
    )
    out.to_csv(os.path.join(RESULTS_DIR, "publication_metrics_table.csv"), index=False)
    return out


def save_pairwise_conclusions(tests_df, results_dir):
    lines = []
    lines.append("Comparative conclusions restated from paired tests")
    lines.append("")
    lines.append(
        "Architecture comparisons were interpreted with paired tests on the same "
        "test set: exact McNemar for overall accuracy, and DeLong one-vs-rest for "
        "AUC per class. P-values were adjusted with Holm-Bonferroni within each "
        "test family."
    )
    lines.append("")

    for _, row in tests_df.iterrows():
        sig = "statistically significant difference" if row["reject_holm"] else "no statistically significant difference"
        lines.append(
            f"- {row['comparison']} | {row['test_family']} | class={row['class']}: "
            f"{sig} after Holm-Bonferroni (p={row['p_value']:.4g}, p_adj={row['p_holm']:.4g}). "
            f"Detail: {row['detail']}."
        )

    lines.append("")
    lines.append(
        "Superiority claims should therefore only be made when the corresponding "
        "paired comparison remains significant after Holm-Bonferroni. When the "
        "adjusted p-value is not significant, report the numerically better model "
        "without claiming statistically significant superiority."
    )

    out_path = os.path.join(results_dir, "pairwise_conclusions_holm.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _summary_value(summary_df, model_name, metric, column):
    sub = summary_df[(summary_df["Model"] == model_name) & (summary_df["metric"] == metric)]
    if len(sub) == 0:
        return np.nan
    return sub.iloc[0][column]


def _seedavg_ci(seedavg_ci_df, model_name, metric):
    sub = seedavg_ci_df[(seedavg_ci_df["Model"] == model_name) & (seedavg_ci_df["metric"] == metric)]
    if len(sub) == 0:
        return np.nan, np.nan
    return sub.iloc[0]["ci_low"], sub.iloc[0]["ci_high"]


def fmt_metric_for_text(summary_df, seedavg_ci_df, model_name, metric):
    mean = _summary_value(summary_df, model_name, metric, "mean")
    sd = _summary_value(summary_df, model_name, metric, "sd")
    ci_low, ci_high = _seedavg_ci(seedavg_ci_df, model_name, metric)
    return f"{mean:.3f} +/- {sd:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f})"


def save_abstract_and_conclusions_text(summary_df, seedavg_ci_df, tests_df, results_dir):
    if summary_df.empty:
        return

    f1_rows = summary_df[summary_df["metric"] == "f1_macro"].copy()
    if f1_rows.empty:
        return

    best_model = f1_rows.sort_values("mean", ascending=False).iloc[0]["Model"]
    n_seeds = int(f1_rows["n_seeds"].max())

    lines = []
    lines.append("Suggested text for abstract and conclusions")
    lines.append("")
    lines.append(
        f"Training and evaluation were repeated with {n_seeds} seeds per architecture "
        f"({', '.join(map(str, RUN_SEEDS))}). The best mean F1-macro corresponded to "
        f"{best_model}: {fmt_metric_for_text(summary_df, seedavg_ci_df, best_model, 'f1_macro')}. "
        f"Its accuracy was {fmt_metric_for_text(summary_df, seedavg_ci_df, best_model, 'accuracy')} "
        f"and its macro AUC was {fmt_metric_for_text(summary_df, seedavg_ci_df, best_model, 'auc_macro')}."
    )

    # XAI / pointing game / energy fraction summary, if available.
    if "pointing_game_acc" in summary_df["metric"].unique():
        lines.append("")
        lines.append(
            f"In the explainability analysis (Integrated Gradients, target=predicted class), "
            f"{best_model} obtained a Pointing Game Accuracy of "
            f"{fmt_metric_for_text(summary_df, seedavg_ci_df, best_model, 'pointing_game_acc')}, "
            f"a Trustworthy Pointing Game Accuracy of "
            f"{fmt_metric_for_text(summary_df, seedavg_ci_df, best_model, 'trustworthy_pg_acc')}, "
            f"and a mean energy fraction of "
            f"{fmt_metric_for_text(summary_df, seedavg_ci_df, best_model, 'mean_energy_fraction')}. "
            f"The mean Dice score between thresholded attribution and the mask was "
            f"{fmt_metric_for_text(summary_df, seedavg_ci_df, best_model, 'mean_xai_dice')}."
        )

    lines.append("")
    lines.append(
        "Model comparisons were interpreted using exact McNemar for overall accuracy "
        "and DeLong one-vs-rest for AUC per class, with Holm-Bonferroni correction."
    )
    lines.append("")

    for _, row in tests_df.iterrows():
        if row["reject_holm"]:
            lines.append(
                f"- {row['comparison']} showed a significant difference in {row['test_family']} "
                f"(class={row['class']}; p_adj={row['p_holm']:.4g})."
            )
        else:
            lines.append(
                f"- {row['comparison']} showed no significant difference in {row['test_family']} "
                f"(class={row['class']}; p_adj={row['p_holm']:.4g})."
            )

    lines.append("")
    lines.append(
        "Recommended phrasing: claim superiority only where p_adj < 0.05. When a "
        "model has a better mean but p_adj >= 0.05, describe it as a numerical "
        "advantage without sufficient statistical evidence of superiority."
    )

    out_path = os.path.join(results_dir, "abstract_and_conclusions_bootstrap_paired.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_final_statistics(models, seeds):
    metrics_by_seed, summary_df = summarize_seed_runs(models, seeds)
    if not metrics_by_seed.empty and "training_time_sec" in metrics_by_seed.columns:
        training_summary = (
            metrics_by_seed.groupby("Model")["training_time_sec"]
            .agg(["count", "mean", "std"])
            .reset_index()
            .rename(columns={"count": "n_seeds", "mean": "mean_training_time_sec", "std": "sd_training_time_sec"})
        )
        training_summary["mean_training_time_min"] = training_summary["mean_training_time_sec"] / 60
        training_summary.to_csv(os.path.join(RESULTS_DIR, "training_time_summary.csv"), index=False)
    aggregated_predictions = load_seed_averaged_predictions(models, seeds)
    seedavg_ci_df = save_seed_averaged_prediction_cis(aggregated_predictions)
    plot_seed_averaged_roc_curves(aggregated_predictions, CLASS_NAMES, RESULTS_DIR)
    plot_seed_averaged_pr_curves(aggregated_predictions, CLASS_NAMES, RESULTS_DIR)

    pointing_records_by_model = load_seed_averaged_pointing_records(models, seeds)
    pointing_seedavg_ci_df = save_seed_averaged_pointing_cis(pointing_records_by_model)
    if not pointing_seedavg_ci_df.empty:
        common_cols = [c for c in seedavg_ci_df.columns if c in pointing_seedavg_ci_df.columns]
        seedavg_ci_df = pd.concat(
            [seedavg_ci_df, pointing_seedavg_ci_df[common_cols]],
            ignore_index=True,
        ) if len(seedavg_ci_df) else pointing_seedavg_ci_df[common_cols]

    publication_df = make_publication_table(summary_df, seedavg_ci_df)

    tests_df = paired_model_tests(aggregated_predictions, CLASS_NAMES, alpha=ALPHA)
    tests_path = os.path.join(RESULTS_DIR, "paired_tests_mcnemar_delong_holm.csv")
    tests_df.to_csv(tests_path, index=False)

    save_pairwise_conclusions(tests_df, RESULTS_DIR)
    save_abstract_and_conclusions_text(summary_df, seedavg_ci_df, tests_df, RESULTS_DIR)

    print(f"Per-seed metrics: {os.path.join(RESULTS_DIR, 'metrics_by_seed.csv')}")
    print(f"Mean +/- SD: {os.path.join(RESULTS_DIR, 'metrics_mean_sd.csv')}")
    print(f"Bootstrap CI: {os.path.join(RESULTS_DIR, 'metrics_seed_averaged_bootstrap_ci.csv')}")
    print(f"XAI bootstrap CI: {os.path.join(RESULTS_DIR, 'pointing_game_seed_averaged_bootstrap_ci.csv')}")
    print(f"Seed-averaged ROC: {os.path.join(RESULTS_DIR, 'seed_averaged_roc')}")
    print(f"Seed-averaged PR: {os.path.join(RESULTS_DIR, 'seed_averaged_pr')}")
    print(f"Training times: {os.path.join(RESULTS_DIR, 'training_time_summary.csv')}")
    print(f"Environment: {os.path.join(RESULTS_DIR, 'environment_summary.csv')}")
    print(f"Publication table: {os.path.join(RESULTS_DIR, 'publication_metrics_table.csv')}")
    print(f"Paired tests: {tests_path}")
    print(f"Final text: {os.path.join(RESULTS_DIR, 'abstract_and_conclusions_bootstrap_paired.txt')}")

    return metrics_by_seed, summary_df, seedavg_ci_df, tests_df, publication_df


# EXECUTION
models = ["mixer_b16_224", "resnet50", "efficientnet_b0"]


def refresh_seed_state(seed):
    global RUN_SEED, SEED, train_sampler
    RUN_SEED = seed
    SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_seed(seed)
    train_sampler = build_train_sampler(seed)
    log_reproducibility_info(RESULTS_DIR)


def run_one_seed(seed):
    refresh_seed_state(seed)
    all_results = {}

    for model_name in models:
        print(f"\n>>> MODEL {model_name} | SEED {seed} <<<")
        params = load_or_optimize_params(model_name, seed)
        res = train_final_model(model_name, params, run_seed=seed)
        all_results[model_name] = res

        metrics = compute_clinical_metrics(res["y_true"], res["y_pred"], res["y_prob"], CLASS_NAMES)
        metrics["test_loss"] = res["test_loss"]
        metrics["training_time_sec"] = res["training_time_sec"]
        metrics["epochs_trained"] = res["epochs_trained"]
        res.update(metrics)

        pg_records_df = pd.DataFrame()
        if RUN_XAI_POINTING_GAME:
            checkpoint_path = res["checkpoint_path"]
            xai_model = create_timm_model(
                model_name,
                pretrained=False,
                num_classes=3,
                drop_rate=res["best_params"]["dropout"],
            ).to(DEVICE)
            xai_model.load_state_dict(safe_torch_load(checkpoint_path, map_location=DEVICE))

            pg_res, pg_records_df = evaluate_all_heatmaps_pointing_game(
                xai_model,
                model_name,
                test_dataset,
                DEVICE,
                limit=None,
                return_records=True,
            )
            metrics.update(pg_res)
            res.update(pg_res)

        cls_ci_df = bootstrap_classification_metric_ci(
            res["y_true"],
            res["y_pred"],
            res["y_prob"],
            CLASS_NAMES,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + seed,
            alpha=ALPHA,
        )
        pg_ci_df = bootstrap_pointing_metric_ci(
            pg_records_df,
            CLASS_NAMES,
            n_boot=BOOTSTRAP_REPLICATES,
            seed=BOOTSTRAP_SEED + seed + 5000,
            alpha=ALPHA,
        )
        metrics_ci_df = pd.concat([cls_ci_df, pg_ci_df], ignore_index=True)

        run_dir = model_run_dir(RESULTS_DIR, model_name, seed)
        save_prediction_artifacts(res, model_name, seed, RESULTS_DIR)
        save_metrics_artifacts(metrics, metrics_ci_df, model_name, seed, RESULTS_DIR)
        plot_training_results(res["history"], model_name, run_dir)
        plot_confusion_matrix_heatmap(res["cm"], CLASS_NAMES, model_name, run_dir)
        plot_roc_curves(res["y_true"], res["y_prob"], CLASS_NAMES, model_name, run_dir)
        plot_precision_recall_curves(res["y_true"], res["y_prob"], CLASS_NAMES, model_name, run_dir)

        if RUN_XAI_POINTING_GAME and len(pg_records_df) > 0:
            pg_records_df.to_csv(os.path.join(run_dir, "pointing_game_records.csv"), index=False)

        if RUN_XAI_POINTING_GAME and SAVE_EXAMPLE_HEATMAPS:
            plot_model_heatmaps(xai_model, model_name, test_dataset, DEVICE, results_dir=run_dir)

        if RUN_XAI_POINTING_GAME and SAVE_ALL_HEATMAPS:
            save_all_test_heatmaps(xai_model, model_name, test_dataset, DEVICE, results_dir=run_dir, limit=None)

    seed_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Model": model_name,
                "seed": seed,
                "training_time_sec": res.get("training_time_sec", np.nan),
                "training_time_min": res.get("training_time_sec", np.nan) / 60,
                "epochs_trained": res.get("epochs_trained", np.nan),
            }
            for model_name, res in all_results.items()
        ]
    ).to_csv(os.path.join(seed_dir, "training_times.csv"), index=False)
    generate_global_comparison(all_results, seed_dir)
    plot_overfitting_analysis(all_results, seed_dir)
    print(f"\n[SEED COMPLETE] seed={seed} | Results in: {RESULTS_DIR}")
    return all_results


if RUN_MODE == "manual_seed":
    run_one_seed(RUN_SEED)
elif RUN_MODE == "all_seeds":
    for seed_value in RUN_SEEDS:
        run_one_seed(seed_value)
    run_final_statistics(models, RUN_SEEDS)
elif RUN_MODE == "final_statistics_only":
    run_final_statistics(models, RUN_SEEDS)
else:
    raise ValueError(f"Unrecognized RUN_MODE: {RUN_MODE}")
