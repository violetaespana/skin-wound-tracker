"""
Módulo PyTorch Dataset & DataLoaders para el Proyecto Final de Visión Computacional
(Opción B: Heridas Crónicas / Clínicas — Clasificación Multiclase + Segmentación de Área).

Uso:
    python scripts/wound_dataset.py
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = PROJECT_ROOT / "dataset_unified"
SEG_DIR = PROJECT_ROOT / "dataset_segmentation"
META_DIR = PROJECT_ROOT / "metadata"

CANONICAL_CLASSES = [
    "diabetic_foot_ulcer",
    "pressure_ulcer",
    "venous_ulcer",
    "surgical_wound",
    "normal_skin",
]
CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(CANONICAL_CLASSES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_classification_transforms(img_size: int = 224, is_train: bool = True) -> transforms.Compose:
    """Transformaciones estándar para CNN / ConvNeXt / ViT (sin data leakage)."""
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ChronicWoundDataset(Dataset):
    """Dataset de clasificación de 5 clases clínicas de heridas crónicas y piel sana."""

    def __init__(
        self,
        root_dir: Path | str = UNIFIED_DIR,
        split: str = "train",
        transform: Optional[Callable] = None,
        img_size: int = 224,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        if not self.split_dir.exists():
            raise FileNotFoundError(f"No existe el directorio del split: {self.split_dir}")

        self.transform = transform or get_classification_transforms(img_size=img_size, is_train=(split == "train"))
        self.samples: List[Tuple[Path, int]] = []
        self.targets: List[int] = []

        for cls_name in CANONICAL_CLASSES:
            cls_dir = self.split_dir / cls_name
            if not cls_dir.exists():
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((img_path, cls_idx))
                    self.targets.append(cls_idx)

        if not self.samples:
            raise RuntimeError(f"No se encontraron imágenes válidas en {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, target

    def get_class_counts(self) -> Dict[str, int]:
        counts = Counter(self.targets)
        return {IDX_TO_CLASS[i]: counts.get(i, 0) for i in range(len(CANONICAL_CLASSES))}

    def get_class_weights(self) -> torch.Tensor:
        """Pesos inversamente proporcionales a la frecuencia para CrossEntropyLoss / FocalLoss."""
        counts = Counter(self.targets)
        total = len(self.targets)
        num_classes = len(CANONICAL_CLASSES)
        weights = []
        for i in range(num_classes):
            c = max(1, counts.get(i, 0))
            weights.append(total / (num_classes * c))
        return torch.tensor(weights, dtype=torch.float32)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Muestreador balanceado por clase para entrenamiento."""
        class_w = self.get_class_weights()
        sample_weights = torch.tensor([class_w[t] for t in self.targets], dtype=torch.double)
        return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


class WoundSegmentationDataset(Dataset):
    """Dataset de pares (imagen clínica, máscara binaria) para medición y seguimiento de área de úlcera."""

    def __init__(
        self,
        project_root: Path | str = PROJECT_ROOT,
        manifest_csv: Optional[Path | str] = None,
        split: Optional[str] = None,
        img_size: int = 256,
        augment: bool = False,
    ) -> None:
        self.project_root = Path(project_root)
        self.manifest_csv = Path(manifest_csv) if manifest_csv else (self.project_root / "metadata" / "manifest_segmentation.csv")
        self.img_size = img_size
        self.augment = augment
        self.pairs: List[Tuple[Path, Path, str]] = []

        if self.manifest_csv.exists():
            with open(self.manifest_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if split is not None and row["split"] != split:
                        continue
                    img_p = self.project_root / row["image_rel_path"]
                    msk_p = self.project_root / row["mask_rel_path"]
                    if img_p.exists() and msk_p.exists():
                        self.pairs.append((img_p, msk_p, row["source"]))
        else:
            img_dir = self.project_root / "dataset_segmentation" / "images"
            msk_dir = self.project_root / "dataset_segmentation" / "masks"
            for img_p in sorted(img_dir.glob("*.png")):
                msk_p = msk_dir / img_p.name
                if msk_p.exists():
                    self.pairs.append((img_p, msk_p, "unknown"))

        if not self.pairs:
            raise RuntimeError(f"No se encontraron pares imagen/máscara de segmentación (split={split}).")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_p, msk_p, _ = self.pairs[idx]
        with Image.open(img_p) as im, Image.open(msk_p) as mk:
            rgb = im.convert("RGB").resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            mask = mk.convert("L").resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

            if self.augment:
                if torch.rand(1).item() > 0.5:
                    rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if torch.rand(1).item() > 0.5:
                    rgb = rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                    mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            img_t = self.normalize(self.to_tensor(rgb))
            mask_arr = (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)
            mask_t = torch.from_numpy(mask_arr).unsqueeze(0)
        return img_t, mask_t


def create_dataloaders(
    batch_size: int = 32,
    img_size: int = 224,
    seg_size: int = 256,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
) -> Dict[str, object]:
    """Construye y retorna los DataLoaders listos para entrenamiento y evaluación."""
    train_ds = ChronicWoundDataset(UNIFIED_DIR, split="train", img_size=img_size)
    val_ds = ChronicWoundDataset(UNIFIED_DIR, split="val", img_size=img_size)
    test_ds = ChronicWoundDataset(UNIFIED_DIR, split="test", img_size=img_size)

    sampler = train_ds.get_weighted_sampler() if use_weighted_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    seg_train_ds = WoundSegmentationDataset(PROJECT_ROOT, split="train", img_size=seg_size, augment=True)
    seg_val_ds = WoundSegmentationDataset(PROJECT_ROOT, split="val", img_size=seg_size, augment=False)
    seg_train_loader = DataLoader(seg_train_ds, batch_size=min(16, batch_size), shuffle=True, num_workers=num_workers)
    seg_val_loader = DataLoader(seg_val_ds, batch_size=min(16, batch_size), shuffle=False, num_workers=num_workers)

    return {
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_weights": train_ds.get_class_weights(),
        "seg_train_dataset": seg_train_ds,
        "seg_val_dataset": seg_val_ds,
        "seg_train_loader": seg_train_loader,
        "seg_val_loader": seg_val_loader,
    }


if __name__ == "__main__":
    bundle = create_dataloaders(batch_size=16, img_size=224, seg_size=256, num_workers=0)
    print("=== Prueba de Humo: ChronicWoundDataset (5 clases) ===")
    for sp in ["train", "val", "test"]:
        ds: ChronicWoundDataset = bundle[f"{sp}_dataset"]
        print(f"  Split {sp:5s}: {len(ds):4d} imgs | {ds.get_class_counts()}")
    print("  Class weights:", [round(float(w), 4) for w in bundle["class_weights"]])

    xb, yb = next(iter(bundle["train_loader"]))
    print(f"  Train batch -> x: {tuple(xb.shape)} {xb.dtype}, y: {tuple(yb.shape)} labels={sorted(set(yb.tolist()))}")
    x_val, y_val = next(iter(bundle["val_loader"]))
    print(f"  Val batch   -> x: {tuple(x_val.shape)}, y: {tuple(y_val.shape)}")
    x_tst, y_tst = next(iter(bundle["test_loader"]))
    print(f"  Test batch  -> x: {tuple(x_tst.shape)}, y: {tuple(y_tst.shape)}")

    print("\n=== Prueba de Humo: WoundSegmentationDataset ===")
    print(f"  Seg Train: {len(bundle['seg_train_dataset'])} pares | Seg Val: {len(bundle['seg_val_dataset'])} pares")
    sx, sm = next(iter(bundle["seg_train_loader"]))
    print(f"  Seg batch -> image: {tuple(sx.shape)} {sx.dtype}, mask: {tuple(sm.shape)} {sm.dtype}, unique_mask_vals={sorted(torch.unique(sm).tolist())}")
    print("\n[OK] Todos los DataLoaders funcionan correctamente.")
