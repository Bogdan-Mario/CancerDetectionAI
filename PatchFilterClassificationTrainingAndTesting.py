import os
import random
import glob
import json
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_patches_from_dataset(img_dir, mask_dir, patch_size, overlap, balance_mode):
    img_paths = glob.glob(os.path.join(img_dir, "*.png"))
    patches_dataset = []
    stride = patch_size // 2 if overlap else patch_size

    print(f"\nextracting patches: size {patch_size}, overlap {overlap}, mode {balance_mode}")

    for img_path in img_paths:
        base_name = os.path.basename(img_path).replace(".png", "")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        mask_mass = np.zeros((h, w), dtype=np.uint8)
        mask_calc = np.zeros((h, w), dtype=np.uint8)

        mass_path = os.path.join(mask_dir, "Mass_Only", f"{base_name}_mass.png")
        calc_path = os.path.join(mask_dir, "Calcification_Only", f"{base_name}_calc.png")

        if os.path.exists(mass_path): mask_mass = cv2.imread(mass_path, cv2.IMREAD_GRAYSCALE)
        if os.path.exists(calc_path): mask_calc = cv2.imread(calc_path, cv2.IMREAD_GRAYSCALE)

        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask_mass_padded = cv2.copyMakeBorder(mask_mass, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        mask_calc_padded = cv2.copyMakeBorder(mask_calc, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        ph, pw = img_padded.shape

        for y in range(0, ph - patch_size + 1, stride):
            for x in range(0, pw - patch_size + 1, stride):
                patch_img = img_padded[y:y + patch_size, x:x + patch_size]

                if np.mean(patch_img) < 5:  
                    continue

                pmass = mask_mass_padded[y:y + patch_size, x:x + patch_size]
                pcalc = mask_calc_padded[y:y + patch_size, x:x + patch_size]

                mass_present = np.any(pmass > 127)
                calc_present = np.any(pcalc > 127)

                if mass_present and calc_present:
                    label = 3
                elif calc_present:
                    label = 2
                elif mass_present:
                    label = 1
                else:
                    label = 0

                patches_dataset.append({'patch': patch_img, 'label': label})

    healthy_patches = [p for p in patches_dataset if p['label'] == 0]
    sick_patches = [p for p in patches_dataset if p['label'] > 0]

    if balance_mode == 'sick_only':
        final_dataset = [{'patch': p['patch'], 'label': p['label'] - 1} for p in sick_patches]
        num_classes = 3

        # Diagnostic Print
        c0 = sum(1 for p in final_dataset if p['label'] == 0)
        c1 = sum(1 for p in final_dataset if p['label'] == 1)
        c2 = sum(1 for p in final_dataset if p['label'] == 2)
        print(f"  dataset: mass {c0}, calc {c1}, both {c2}")

    else:  # Balanced Mode
        if len(healthy_patches) > len(sick_patches) and len(sick_patches) > 0:
            healthy_patches = random.sample(healthy_patches, len(sick_patches))
        final_dataset = healthy_patches + sick_patches
        num_classes = 4

        # Diagnostic Print
        c0 = sum(1 for p in final_dataset if p['label'] == 0)
        c1 = sum(1 for p in final_dataset if p['label'] == 1)
        c2 = sum(1 for p in final_dataset if p['label'] == 2)
        c3 = sum(1 for p in final_dataset if p['label'] == 3)
        print(f"  dataset: healthy {c0}, mass {c1}, calc {c2}, both {c3}")

    print(f"  total: {len(final_dataset)} patches, {num_classes} classes")
    return final_dataset, num_classes
# Dataset loader and custom loss functions
class MammographyPatchDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        patch_3ch = cv2.cvtColor(item['patch'], cv2.COLOR_GRAY2RGB)
        if self.transform:
            augmented = self.transform(image=patch_3ch)
            patch_3ch = augmented['image']
        else:
            patch_3ch = patch_3ch.astype(np.float32) / 255.0
            patch_3ch = np.transpose(patch_3ch, (2, 0, 1))
            patch_3ch = torch.tensor(patch_3ch, dtype=torch.float32)
        return patch_3ch, torch.tensor(item['label'], dtype=torch.long)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


class DiceBCEClassificationLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceBCEClassificationLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets)
        probs = F.softmax(inputs, dim=1)
        one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        intersection = (probs * one_hot).sum(dim=0)
        cardinality = probs.sum(dim=0) + one_hot.sum(dim=0)
        dice_loss = 1 - ((2. * intersection + self.smooth) / (cardinality + self.smooth)).mean()
        return ce + dice_loss


# Load model from timm or open_clip
def get_model(model_name, num_classes):
    if model_name == "resnet50":
        return timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
    elif model_name == "medclip":
        try:
            import open_clip
            model, _, _ = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', pretrained=True
            )
            model.visual.output_tokens = False
            embed_dim = 512

            return nn.Sequential(model.visual, nn.Linear(embed_dim, num_classes))

        except Exception as e:
            print(f"biomedclip failed: {e}, using standard vit")
            return timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError("Unknown model identifier")

# Run a single experiment with the given config
def run_experiment(config, train_data, val_data, num_classes, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)

    # If the model is MedCLIP (ViT), we must upscale the patch to 224x224
    # If it is ResNet, we keep the original extracted patch size 
    is_vit = (config['model'] == 'medclip')

    aug_list_train = []
    aug_list_val = []

    if is_vit:
        aug_list_train.append(A.Resize(224, 224))
        aug_list_val.append(A.Resize(224, 224))

    # Configure targeted augmentations
    if config['aug'] == 'clahe_only':
        aug_list_train.extend([A.CLAHE(p=1.0), A.Normalize(), ToTensorV2()])
    else:  # Heavy Medical Combo
        aug_list_train.extend([
            A.CLAHE(p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.3),
            A.Normalize(),
            ToTensorV2()
        ])

    aug_list_val.extend([A.CLAHE(p=1.0), A.Normalize(), ToTensorV2()])

    train_transform = A.Compose(aug_list_train)
    val_transform = A.Compose(aug_list_val)

    train_loader = DataLoader(MammographyPatchDataset(train_data, train_transform), batch_size=64, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(MammographyPatchDataset(val_data, val_transform), batch_size=64, shuffle=False,
                            num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config['model'], num_classes).to(device)

    criterion = FocalLoss() if config['loss'] == 'focal' else DiceBCEClassificationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < 3: return float(epoch + 1) / 3.0
        return 0.9 ** (epoch - 3)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_loss = float('inf')
    early_stop_counter, patience = 0, 5

    metrics_history_log = []
    history_train_loss, history_val_loss, history_val_acc = [], [], []

    for epoch in range(300):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_train_loss = train_loss / total
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        history_train_loss.append(epoch_train_loss)
        history_val_loss.append(epoch_val_loss)
        history_val_acc.append(epoch_val_acc)

        metrics_history_log.append(
            f"Epoch {epoch}: Train Loss {epoch_train_loss:.4f} | Val Loss {epoch_val_loss:.4f} | Val Acc {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            early_stop_counter = 0

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix (Acc: {epoch_val_acc:.2f})")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(exp_dir, "confusion_matrix.png"))
            plt.close()

            report_text = classification_report(all_labels, all_preds, digits=4, zero_division=0)
            report_dict = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)

            with open(os.path.join(exp_dir, "full_classification_report.txt"), "w") as f:
                f.write(report_text)
            with open(os.path.join(exp_dir, "full_classification_metrics.json"), "w") as f:
                json.dump(report_dict, f, indent=4)
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            metrics_history_log.append("Early stopping triggered.")
            break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_train_loss, label='Train Loss', color='blue', marker='o', markersize=4)
    plt.plot(history_val_loss, label='Val Loss', color='red', marker='o', markersize=4)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history_val_acc, label='Val Accuracy', color='green', marker='o', markersize=4)
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "training_history_plots.png"), dpi=300)
    plt.close()

    with open(os.path.join(exp_dir, "experiment_details.txt"), "w") as f:
        f.write(f"CONFIGURATION:\n{json.dumps(config, indent=4)}\n\nLOG:\n" + "\n".join(metrics_history_log))
if __name__ == "__main__":
    seed_everything(42)

    IMAGE_DIR = "/Normal"
    MASK_DIR = "/Masks"
    OUTPUT_BASE_DIR = "./Optimized_Experiment_Results_300_Epochs"

    patch_sizes = [32, 64, 128, 256]
    balance_modes = ['balanced', 'sick_only']
    models = ['resnet50', 'medclip']
    losses = ['dice_bce', 'focal']
    augmentations = ['clahe_only', 'medical_heavy']

    print(f"starting: 32 experiments")

    for p_size in patch_sizes:
        for b_mode in balance_modes:


            tier_fully_computed = True
            for model_n in models:
                for loss_type in losses:
                    for aug_type in augmentations:
                        exp_name = f"sz{p_size}_{b_mode}_{model_n}_{loss_type}_{aug_type}"
                        exp_dir = os.path.join(OUTPUT_BASE_DIR, exp_name)
                        if not os.path.exists(os.path.join(exp_dir, "full_classification_metrics.json")):
                            tier_fully_computed = False
                            break

            if tier_fully_computed:
                print(f"  skipping size {p_size} ({b_mode}): all done")
                continue

            patches, num_cls = extract_patches_from_dataset(IMAGE_DIR, MASK_DIR, p_size, overlap=True,
                                                            balance_mode=b_mode)
            if len(patches) < 15:
                continue

            train_data, val_data = train_test_split(patches, test_size=0.2, random_state=42)

            for model_n in models:
                for loss_type in losses:
                    for aug_type in augmentations:

                        exp_name = f"sz{p_size}_{b_mode}_{model_n}_{loss_type}_{aug_type}"
                        exp_dir = os.path.join(OUTPUT_BASE_DIR, exp_name)

                        if os.path.exists(os.path.join(exp_dir, "full_classification_metrics.json")):
                            print(f"  skipped: {exp_name} (done)")
                            continue

                        config_item = {
                            "patch_size": p_size, "overlap": True, "balance_mode": b_mode,
                            "model": model_n, "pretrained": True, "lr": 3e-4, "loss": loss_type, "aug": aug_type
                        }

                        print(f"  running: {exp_name}")
                        run_experiment(config_item, train_data, val_data, num_cls, exp_dir)
