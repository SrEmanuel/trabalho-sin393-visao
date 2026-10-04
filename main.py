import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

MODEL_NAME = "mobilenet_v3_small"  # modelos disponiveis: "resnet18" e "mobilenet_v3_small"
DATASET_DIR = Path("datasets/")
RESULTS_DIR = Path("results")

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5

USE_AUGMENTATION = True
USE_CLASS_WEIGHTS = True # compensa o desbalanceamento das classes na loss

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

N_EXAMPLES_QUALITATIVE = 5 # numero de exemplos certos/errados na analise qualitativa

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train):
    if train and USE_AUGMENTATION:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label


def build_model(name, num_classes):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Modelo '{name}' nao suportado.")
    return model


def make_dataloader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train_model(model, train_loader, valid_loader, criterion, device, save_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_loader(model, valid_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoca {epoch:02d}/{EPOCHS} | "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_dir / "best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"Early stopping na epoca {epoch}")
                break

    return history


def plot_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

@torch.no_grad()
def get_predictions(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(1).cpu().tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de Confusao - {MODEL_NAME}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def qualitative_analysis(model, test_dataset, class_names, device, save_path):
    examples_correct, examples_wrong = [], []

    indices = list(range(len(test_dataset)))
    for idx in indices:
        image, label = test_dataset[idx]
        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device)).argmax(1).item()
        if pred == label:
            examples_correct.append((image, label, pred))
        else:
            examples_wrong.append((image, label, pred))
        if len(examples_correct) >= N_EXAMPLES_QUALITATIVE and len(examples_wrong) >= N_EXAMPLES_QUALITATIVE:
            break

    n_correct = min(N_EXAMPLES_QUALITATIVE, len(examples_correct))
    n_wrong = min(N_EXAMPLES_QUALITATIVE, len(examples_wrong))
    fig, axes = plt.subplots(2, N_EXAMPLES_QUALITATIVE, figsize=(2.4 * N_EXAMPLES_QUALITATIVE, 5))

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for col, (image, label, pred) in enumerate(examples_correct[:N_EXAMPLES_QUALITATIVE]):
        img = (image * std + mean).permute(1, 2, 0).numpy()
        axes[0, col].imshow(np.clip(img, 0, 1))
        axes[0, col].set_title(f"Real: {class_names[label]}\nPredito: {class_names[pred]}", color="green", fontsize=9)
        axes[0, col].axis("off")

    for col, (image, label, pred) in enumerate(examples_wrong[:N_EXAMPLES_QUALITATIVE]):
        img = (image * std + mean).permute(1, 2, 0).numpy()
        axes[1, col].imshow(np.clip(img, 0, 1))
        axes[1, col].set_title(f"Real: {class_names[label]}\nPredito: {class_names[pred]}", color="red", fontsize=9)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Corretas", fontsize=10)
    axes[1, 0].set_ylabel("Incorretas", fontsize=10)
    fig.suptitle(f"Analise Qualitativa - {MODEL_NAME} (topo: acertos, baixo: erros)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Dispositivo: {device}")
    print(f"Modelo: {MODEL_NAME}")

    save_dir = RESULTS_DIR / MODEL_NAME
    save_dir.mkdir(parents=True, exist_ok=True)

    # Carregar o dataset e dividir em treino/validacao/teste
    
    full_dataset = datasets.ImageFolder(DATASET_DIR)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {class_names}")

    labels = np.array(full_dataset.targets)
    indices = np.arange(len(labels))
    counts = pd.Series(labels).value_counts().sort_index()
    print("\nDistribuicao das classes no dataset completo:")
    for i, count in enumerate(counts):
        print(f"  {class_names[i]}: {count}")

    train_idx, temp_idx = train_test_split(
        indices, test_size=VALID_RATIO + TEST_RATIO,
        stratify=labels, random_state=SEED)
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=TEST_RATIO / (VALID_RATIO + TEST_RATIO),
        stratify=labels[temp_idx], random_state=SEED)

    split_df = pd.DataFrame({
        "path": [full_dataset.samples[i][0] for i in indices],
        "label": [class_names[y] for y in labels],
        "split": ["train" if i in train_idx else "valid" if i in valid_idx else "test" for i in indices],
    })
    split_df.to_csv(RESULTS_DIR / "split.csv", index=False)

    print(f"\nDivisao (seed={SEED}, estratificada): "
          f"treino={len(train_idx)}, validacao={len(valid_idx)}, teste={len(test_idx)}")

    train_dataset = SubsetWithTransform(Subset(full_dataset, train_idx), get_transforms(train=True))
    valid_dataset = SubsetWithTransform(Subset(full_dataset, valid_idx), get_transforms(train=False))
    test_dataset = SubsetWithTransform(Subset(full_dataset, test_idx), get_transforms(train=False))

    train_loader = make_dataloader(train_dataset, shuffle=True)
    valid_loader = make_dataloader(valid_dataset, shuffle=False)
    test_loader = make_dataloader(test_dataset, shuffle=False)

    # Criar o modelo e a funcao de perda

    model = build_model(MODEL_NAME, num_classes).to(device)

    if USE_CLASS_WEIGHTS:
        train_labels = [full_dataset.targets[i] for i in train_idx]
        weights = 1.0 / np.bincount(train_labels)
        weights = weights / weights.sum() * num_classes
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print("\nPesos das classes na loss:", np.round(weights, 3))
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Treinar com validacao e early stopping

    print(f"\n--- Treinando {MODEL_NAME} ---")
    history = train_model(model, train_loader, valid_loader, criterion, device, save_dir)
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)
    plot_history(history, save_dir / "history.png")


    # Avaliar apenas no conjunto de teste
    model.load_state_dict(torch.load(save_dir / "best_model.pth", weights_only=True))
    y_true, y_pred = get_predictions(model, test_loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    print(f"\n--- Resultados no conjunto de teste ({MODEL_NAME}) ---")
    print(f"Acuracia: {accuracy:.4f}")
    print(f"Precisao (macro): {precision_macro:.4f} | Precisao (weighted): {precision_weighted:.4f}")
    print(f"Recall (macro): {recall_macro:.4f} | Recall (weighted): {recall_weighted:.4f}")
    print(f"F1-score (macro): {f1_macro:.4f} | F1-score (weighted): {f1_weighted:.4f}")
    print("\nRelatorio por classe:")
    print(report)

    with open(save_dir / "classification_report.txt", "w") as f:
        f.write(f"Modelo: {MODEL_NAME}\n")
        f.write(f"Estrategia de agregacao das metricas: macro e weighted "
                f"(weighted e a mais adequada, pois as classes sao desbalanceadas)\n\n")
        f.write(report)

    plot_confusion_matrix(y_true, y_pred, class_names, save_dir / "confusion_matrix.png")
    qualitative_analysis(model, test_dataset, class_names, device, save_dir / "qualitative.png")

    summary = pd.DataFrame([{
        "model": MODEL_NAME,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }])
    summary_path = RESULTS_DIR / "metrics_summary.csv"
    if summary_path.exists():
        summary.to_csv(summary_path, mode="a", header=False, index=False)
    else:
        summary.to_csv(summary_path, index=False)

    print(f"\nResultados salvos em: {save_dir}")


if __name__ == "__main__":
    main()
