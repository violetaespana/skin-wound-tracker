"""
Script reproducible de descarga, auditoría, desduplicación forense (anti-data leakage) e integración
del dataset multiclase de Heridas Crónicas y Clínicas (Opción B — Proyecto Final de Visión Computacional).

Clases canónicas:
    0: diabetic_foot_ulcer  (Úlcera de pie diabético)
    1: pressure_ulcer       (Úlcera por presión / Escara)
    2: venous_ulcer         (Úlcera venosa)
    3: surgical_wound       (Herida quirúrgica)
    4: normal_skin          (Piel sana)
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import random
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw_downloads"
UNIFIED_DIR = PROJECT_ROOT / "dataset_unified"
SEG_DIR = PROJECT_ROOT / "dataset_segmentation"
META_DIR = PROJECT_ROOT / "metadata"
FIG_DIR = PROJECT_ROOT / "figures"

SEED = 42
CANONICAL_CLASSES = [
    "diabetic_foot_ulcer",
    "pressure_ulcer",
    "venous_ulcer",
    "surgical_wound",
    "normal_skin",
]

CLASS_DISPLAY_ES = {
    "diabetic_foot_ulcer": "Úlcera Pie Diabético",
    "pressure_ulcer": "Úlcera por Presión",
    "venous_ulcer": "Úlcera Venosa",
    "surgical_wound": "Herida Quirúrgica",
    "normal_skin": "Piel Sana",
}

SOURCE_PRIORITY = {
    "AZH_UWM": 1,
    "Medetec": 2,
    "SurgWound_OSWUMC": 3,
    "Mendeley_hsj38fwnvr": 4,
    "FUSeg_UWM": 5,
}

SOURCES = [
    {
        "name": "uwm_loc_Train.zip",
        "url": "https://raw.githubusercontent.com/uwm-bigdata/wound-classification-using-images-and-locations/main/dataset/Train.zip",
        "desc": "AZH/UWM Wound Classification (Train)",
    },
    {
        "name": "uwm_loc_Test.zip",
        "url": "https://raw.githubusercontent.com/uwm-bigdata/wound-classification-using-images-and-locations/main/dataset/Test.zip",
        "desc": "AZH/UWM Wound Classification (Test)",
    },
    {
        "name": "uwm_cls_ROI.zip",
        "url": "https://raw.githubusercontent.com/uwm-bigdata/wound_classification/master/dataset/ROI.zip",
        "desc": "AZH/UWM Wound Classification ROI",
    },
    {
        "name": "uwm_cls_original.zip",
        "url": "https://raw.githubusercontent.com/uwm-bigdata/wound_classification/master/dataset/original%20images.zip",
        "desc": "AZH/UWM Wound Classification Original Images",
    },
    {
        "name": "vincenzopalma_wound_cls.zip",
        "url": "https://codeload.github.com/VincenzoPalma/WoundTypeClassifier/zip/refs/heads/master",
        "desc": "AZH/UWM WoundTypeClassifier Splits (Train/Val/Test)",
    },
    {
        "name": "mlaradji_wound_care.zip",
        "url": "https://codeload.github.com/mlaradji/deep-learning-for-wound-care/zip/refs/heads/master",
        "desc": "Medetec Wound Database + Clinical Extended",
    },
    {
        "name": "b1b3k47_wound_cls.zip",
        "url": "https://codeload.github.com/b1b3k47/Wound-Classification/zip/refs/heads/master",
        "desc": "Clinical Multiclass Wound Collection (b1b3k47)",
    },
    {
        "name": "mendeley_hsj38fwnvr.zip",
        "url": "https://data.mendeley.com/public-files/datasets/hsj38fwnvr/files/3ceb3eae-289b-4b01-8950-b6a6e5131179/file_downloaded",
        "desc": "Mendeley Lower Limb and Feet Wound Image Dataset v3",
    },
    {
        "name": "uwm_segmentation_repo.zip",
        "url": "https://codeload.github.com/uwm-bigdata/wound-segmentation/zip/refs/heads/master",
        "desc": "UWM Wound Segmentation (FUSeg + AZH Patches + Medetec)",
    },
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dirs() -> None:
    for d in [RAW_DIR, UNIFIED_DIR, SEG_DIR / "images", SEG_DIR / "masks", META_DIR, FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def download_file(item: Dict[str, str], retries: int = 3) -> Tuple[str, bool, str]:
    dest = RAW_DIR / item["name"]
    if dest.exists() and dest.stat().st_size > 10_000:
        try:
            with zipfile.ZipFile(dest, "r") as zf:
                zf.namelist()[:1]
            return item["name"], True, f"Ya existe ({dest.stat().st_size / (1024*1024):.1f} MB)"
        except zipfile.BadZipFile:
            dest.unlink(missing_ok=True)

    tmp_dest = dest.with_suffix(".part")
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                item["url"],
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                with open(tmp_dest, "wb") as f:
                    shutil.copyfileobj(resp, f, length=1024 * 256)
            with zipfile.ZipFile(tmp_dest, "r") as zf:
                _ = len(zf.namelist())
            tmp_dest.replace(dest)
            return item["name"], True, f"Descargado ({dest.stat().st_size / (1024*1024):.1f} MB)"
        except Exception as exc:
            last_err = str(exc)
            tmp_dest.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    return item["name"], False, f"Error tras {retries} intentos: {last_err}"


SURGWOUND_URLS = {
    "train": "https://huggingface.co/datasets/xuxuxuxuxu/SurgWound/resolve/main/train_report.json",
    "val": "https://huggingface.co/datasets/xuxuxuxuxu/SurgWound/resolve/main/val_report.json",
    "test": "https://huggingface.co/datasets/xuxuxuxuxu/SurgWound/resolve/main/test_report.json",
}


def ensure_surgwound_zip(retries: int = 3) -> Tuple[str, bool, str]:
    """Descarga SurgWound (OSWUMC, npj Digital Medicine 2026) desde Hugging Face y lo empaqueta en ZIP."""
    zip_name = "surgwound_oswumc.zip"
    dest = RAW_DIR / zip_name
    if dest.exists() and dest.stat().st_size > 100_000:
        try:
            with zipfile.ZipFile(dest, "r") as zf:
                entries = zf.namelist()
                if len(entries) >= 600:
                    return zip_name, True, f"Ya existe ({dest.stat().st_size / (1024*1024):.1f} MB, {len(entries)} imgs)"
        except zipfile.BadZipFile:
            dest.unlink(missing_ok=True)

    tmp_dest = dest.with_suffix(".part")
    seen_digests = set()
    used_arcnames = set()
    written = 0

    try:
        with zipfile.ZipFile(tmp_dest, "w", compression=zipfile.ZIP_STORED) as zf:
            for split, url in SURGWOUND_URLS.items():
                raw_bytes = None
                last_err = ""
                for attempt in range(1, retries + 1):
                    try:
                        req = urllib.request.Request(
                            url,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                        )
                        with urllib.request.urlopen(req, timeout=90) as resp:
                            raw_bytes = resp.read()
                        break
                    except Exception as exc:
                        last_err = str(exc)
                        time.sleep(2 * attempt)
                if raw_bytes is None:
                    raise RuntimeError(f"Fallo descargando SurgWound {split}_report.json: {last_err}")

                records = json.loads(raw_bytes.decode("utf-8"))
                for idx, rec in enumerate(records):
                    img_b64 = rec.get("image")
                    if not img_b64 or not isinstance(img_b64, str):
                        continue
                    if img_b64.startswith("data:") and "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(img_b64)
                    digest = hashlib.sha256(img_bytes).hexdigest()
                    if digest in seen_digests:
                        continue
                    seen_digests.add(digest)

                    raw_name = Path(str(rec.get("image_name") or f"{rec.get('id', idx)}.jpg")).name
                    arcname = f"SurgWound/{split}_{raw_name}"
                    if arcname in used_arcnames:
                        stem = Path(raw_name).stem
                        suffix = Path(raw_name).suffix or ".jpg"
                        arcname = f"SurgWound/{split}_{stem}_{idx}{suffix}"
                    used_arcnames.add(arcname)
                    zf.writestr(arcname, img_bytes)
                    written += 1

        with zipfile.ZipFile(tmp_dest, "r") as zf:
            _ = len(zf.namelist())
        tmp_dest.replace(dest)
        return zip_name, True, f"Empaquetado ({dest.stat().st_size / (1024*1024):.1f} MB, {written} imgs únicas)"
    except Exception as exc:
        tmp_dest.unlink(missing_ok=True)
        return zip_name, False, f"Error empaquetando SurgWound: {exc}"


def download_all() -> None:
    print("=== [FASE 1] Descargando archivos fuente verificados ===")
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download_file, item): item for item in SOURCES}
        for fut in as_completed(futures):
            name, ok, msg = fut.result()
            status = "OK" if ok else "ERROR"
            print(f"  [{status}] {name}: {msg}")
            if not ok:
                raise RuntimeError(f"Fallo crítico al descargar {name}: {msg}")

    sw_name, sw_ok, sw_msg = ensure_surgwound_zip()
    sw_status = "OK" if sw_ok else "ERROR"
    print(f"  [{sw_status}] {sw_name}: {sw_msg}")
    if not sw_ok:
        raise RuntimeError(f"Fallo crítico al descargar {sw_name}: {sw_msg}")


def is_valid_image_entry(name: str, zip_name: str = "") -> bool:
    if name.endswith("/") or "__MACOSX" in name or ".ipynb_checkpoints" in name:
        return False
    base = os.path.basename(name)
    if base.startswith(".") or base.startswith("._"):
        return False
    if zip_name == "b1b3k47_wound_cls.zip" and "_original_" in base:
        # Excluir copias sintéticas generadas por Augmentor (evita data leakage entre train y test)
        return False
    return Path(base).suffix.lower() in IMAGE_EXTS


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Calcula hash perceptual dHash de 64 bits."""
    gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val


def hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def inspect_archives() -> None:
    print("\n=== [INSPECCIÓN] Estructura interna de archivos descargados ===")
    for item in SOURCES:
        path = RAW_DIR / item["name"]
        if not path.exists():
            print(f"  [FALTA] {item['name']}")
            continue
        with zipfile.ZipFile(path, "r") as zf:
            img_entries = [n for n in zf.namelist() if is_valid_image_entry(n, item["name"])]
            folder_counts = Counter("/".join(n.split("/")[:-1]) for n in img_entries)
            print(f"\n--- {item['name']} ({len(img_entries)} imágenes) ---")
            for folder, cnt in sorted(folder_counts.items()):
                print(f"    {folder}: {cnt}")


def map_entry_to_record(zip_name: str, entry_path: str) -> Optional[Tuple[str, str]]:
    """
    Mapea una ruta dentro de un ZIP hacia (canonical_class, academic_source).
    Retorna None si la entrada no pertenece a las 5 clases clínicas objetivo.
    """
    parts = [p for p in entry_path.split("/") if p]
    if len(parts) < 2:
        return None
    parent = parts[-2]
    full_lower = entry_path.lower()

    # 1. AZH / UWM archives
    if zip_name in {
        "vincenzopalma_wound_cls.zip",
        "uwm_loc_Train.zip",
        "uwm_loc_Test.zip",
        "uwm_cls_ROI.zip",
        "uwm_cls_original.zip",
    }:
        azh_map = {
            "D": "diabetic_foot_ulcer",
            "P": "pressure_ulcer",
            "V": "venous_ulcer",
            "S": "surgical_wound",
            "N": "normal_skin",
        }
        if parent in azh_map:
            return azh_map[parent], "AZH_UWM"
        return None

    # 2. Medetec (colección clínica curada en mlaradji; se excluyen google-images-medetec-*)
    if zip_name == "mlaradji_wound_care.zip":
        if "/medetec-dataset/" not in full_lower:
            return None
        src = "Medetec"

        p_low = parent.lower()
        if p_low in {"pressure-ulcer-images-a", "pressure-ulcer-images-b", "pressure-ulcer-images"}:
            return "pressure_ulcer", src
        if p_low == "leg-ulcer-images":
            return "venous_ulcer", src
        if p_low in {"foot-ulcers", "toes"}:
            return "diabetic_foot_ulcer", src
        if p_low in {"abdominal-wounds", "orthopaedic-wounds", "pilonidal-sinus"}:
            return "surgical_wound", src
        return None

    # 3. FUSeg MICCAI 2021 (uwm_segmentation_repo.zip) para úlcera de pie diabético
    if zip_name == "uwm_segmentation_repo.zip":
        if "foot ulcer segmentation challenge/train/images/" in full_lower or "foot ulcer segmentation challenge/validation/images/" in full_lower:
            return "diabetic_foot_ulcer", "FUSeg_UWM"
        return None

    # 4. Mendeley Lower Limb and Feet Wound Image Dataset v3
    if zip_name == "mendeley_hsj38fwnvr.zip":
        # Nota: en el archivo original la carpeta de pies sanos se llama 'Nomal/'
        if "/nomal/" in full_lower or full_lower.startswith("nomal/") or "/normal/" in full_lower or full_lower.startswith("normal/"):
            return "normal_skin", "Mendeley_hsj38fwnvr"
        return None

    # 5. SurgWound (Ohio State University Wexner Medical Center, npj Digital Medicine 2026)
    if zip_name == "surgwound_oswumc.zip":
        return "surgical_wound", "SurgWound_OSWUMC"

    return None


def decode_and_hash(raw_bytes: bytes) -> Optional[Dict]:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            im.load()
            rgb = im.convert("RGB")
            w, h = rgb.size
            if w < 32 or h < 32:
                return None
            file_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            pixel_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
            dhash = compute_dhash(rgb)
            dhash_flip = compute_dhash(rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
            return {
                "rgb": rgb,
                "width": w,
                "height": h,
                "file_sha256": file_sha256,
                "pixel_sha256": pixel_sha256,
                "dhash": dhash,
                "dhash_flip": dhash_flip,
            }
    except Exception:
        return None


def build_datasets(max_per_class: int = 500) -> None:
    ensure_dirs()
    rng = random.Random(SEED)

    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for cls in CANONICAL_CLASSES:
            (UNIFIED_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    if SEG_DIR.exists():
        shutil.rmtree(SEG_DIR)
    (SEG_DIR / "images").mkdir(parents=True, exist_ok=True)
    (SEG_DIR / "masks").mkdir(parents=True, exist_ok=True)

    # Orden estricto según prioridad de fuente primaria (vincenzopalma reúne Train+Val+Test de AZH).
    # Nota forense: b1b3k47_wound_cls.zip se excluye de la integración porque nuestra auditoría demostró
    # que 1,984 archivos son copias Augmentor (*_original_*) y 512/516 restantes coinciden por SHA-256
    # con uwm_cls_original.zip (fotos sin recortar de los mismos casos AZH ya incluidos como ROI).
    archive_order = [
        "vincenzopalma_wound_cls.zip",
        "uwm_loc_Train.zip",
        "uwm_loc_Test.zip",
        "uwm_cls_ROI.zip",
        "uwm_cls_original.zip",
        "mlaradji_wound_care.zip",
        "surgwound_oswumc.zip",
        "uwm_segmentation_repo.zip",
        "mendeley_hsj38fwnvr.zip",
    ]

    print("\n=== [FASE 2] Extrayendo, validando y desduplicando imágenes ===")
    raw_candidates: List[Dict] = []

    seen_file_sha = set()
    seen_pixel_sha = set()
    seen_dhashes: List[Tuple[int, str, str]] = []
    azh_seen_cases = set()  # (cls_name, case_id) para evitar duplicar mismo paciente ROI vs original

    dedup_stats = {
        "total_scanned": 0,
        "corrupt_or_too_small": 0,
        "azh_same_case_skipped": 0,
        "exact_byte_duplicates": 0,
        "exact_pixel_duplicates": 0,
        "perceptual_near_duplicates": 0,
        "flipped_duplicates": 0,
        "accepted_unique_pool": 0,
        "by_archive": defaultdict(lambda: {"scanned": 0, "accepted": 0, "dup_exact": 0, "dup_perceptual": 0, "azh_case_skip": 0}),
    }

    staging_dir = PROJECT_ROOT / "_staging_unique"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    azh_secondary_archives = {"uwm_cls_ROI.zip", "uwm_cls_original.zip"}

    for zip_name in archive_order:
        zip_path = RAW_DIR / zip_name
        if not zip_path.exists():
            continue
        print(f"  Procesando {zip_name}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            entries = sorted([n for n in zf.namelist() if is_valid_image_entry(n, zip_name)])
            if zip_name == "mlaradji_wound_care.zip":
                def mlaradji_key(n: str) -> Tuple[int, str]:
                    nl = n.lower()
                    if "/medetec-dataset/" in nl:
                        return (0, n)
                    if "/google-images-medetec-classes/" in nl:
                        return (1, n)
                    return (2, n)
                entries.sort(key=mlaradji_key)

            for entry in entries:
                mapped = map_entry_to_record(zip_name, entry)
                if mapped is None:
                    continue
                cls_name, src_name = mapped

                stem = Path(entry).stem
                case_id = stem.split("_")[0] if src_name == "AZH_UWM" else ""

                dedup_stats["total_scanned"] += 1
                dedup_stats["by_archive"][zip_name]["scanned"] += 1

                # Si es uwm_cls_ROI o uwm_cls_original, descartar casos ya cubiertos en vincenzopalma/uwm_loc
                if zip_name in azh_secondary_archives and (cls_name, case_id) in azh_seen_cases:
                    dedup_stats["azh_same_case_skipped"] += 1
                    dedup_stats["by_archive"][zip_name]["azh_case_skip"] += 1
                    continue

                raw_bytes = zf.read(entry)
                decoded = decode_and_hash(raw_bytes)
                if decoded is None:
                    dedup_stats["corrupt_or_too_small"] += 1
                    continue

                f_sha = decoded["file_sha256"]
                p_sha = decoded["pixel_sha256"]
                dh = decoded["dhash"]
                dh_flip = decoded["dhash_flip"]

                if f_sha in seen_file_sha:
                    dedup_stats["exact_byte_duplicates"] += 1
                    dedup_stats["by_archive"][zip_name]["dup_exact"] += 1
                    continue

                if p_sha in seen_pixel_sha:
                    dedup_stats["exact_pixel_duplicates"] += 1
                    dedup_stats["by_archive"][zip_name]["dup_exact"] += 1
                    continue

                is_near_dup = False
                is_flip_dup = False
                for prev_dh, _, _ in seen_dhashes:
                    if hamming_distance(dh, prev_dh) <= 4:
                        is_near_dup = True
                        break
                    if hamming_distance(dh_flip, prev_dh) <= 3:
                        is_flip_dup = True
                        break

                if is_near_dup:
                    dedup_stats["perceptual_near_duplicates"] += 1
                    dedup_stats["by_archive"][zip_name]["dup_perceptual"] += 1
                    continue
                if is_flip_dup:
                    dedup_stats["flipped_duplicates"] += 1
                    dedup_stats["by_archive"][zip_name]["dup_perceptual"] += 1
                    continue

                seen_file_sha.add(f_sha)
                seen_pixel_sha.add(p_sha)
                seen_dhashes.append((dh, cls_name, src_name))

                uid = f"{cls_name}__{len(raw_candidates):05d}"
                group_id = f"AZH_{cls_name}_{case_id}" if src_name == "AZH_UWM" else uid
                st_path = staging_dir / f"{uid}.jpg"
                decoded["rgb"].save(st_path, format="JPEG", quality=95)

                raw_candidates.append(
                    {
                        "uid": uid,
                        "group_id": group_id,
                        "class": cls_name,
                        "source": src_name,
                        "archive": zip_name,
                        "orig_entry": entry,
                        "width": decoded["width"],
                        "height": decoded["height"],
                        "file_sha256": f_sha,
                        "pixel_sha256": p_sha,
                        "dhash": f"{dh:016x}",
                        "staging_path": st_path,
                    }
                )
                dedup_stats["accepted_unique_pool"] += 1
                dedup_stats["by_archive"][zip_name]["accepted"] += 1

        # Registrar los casos AZH vistos después de procesar cada archivo AZH completo
        if zip_name in {"vincenzopalma_wound_cls.zip", "uwm_loc_Train.zip", "uwm_loc_Test.zip", "uwm_cls_ROI.zip"}:
            for r in raw_candidates:
                if r["source"] == "AZH_UWM":
                    cid = Path(r["orig_entry"]).stem.split("_")[0]
                    azh_seen_cases.add((r["class"], cid))

    by_class: Dict[str, List[Dict]] = defaultdict(list)
    for rec in raw_candidates:
        by_class[rec["class"]].append(rec)

    print("\n=== Pool único desduplicado por clase y fuente ===")
    for cls in CANONICAL_CLASSES:
        items = by_class[cls]
        src_c = Counter(r["source"] for r in items)
        print(f"  {cls:22s}: {len(items):4d} | fuentes={dict(src_c)}")

    print(f"\n=== [FASE 3] Balanceo (tope {max_per_class}/clase) y Split Estratificado por Paciente 70/15/15 ===")
    selected_records: List[Dict] = []

    for cls in CANONICAL_CLASSES:
        items = list(by_class[cls])
        # Agrupar por group_id para mantener todos los ROIs de un mismo paciente AZH juntos
        groups_by_prio: Dict[int, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        for it in items:
            prio = SOURCE_PRIORITY.get(it["source"], 99)
            groups_by_prio[prio][it["group_id"]].append(it)

        chosen: List[Dict] = []
        if len(items) > max_per_class:
            for prio in sorted(groups_by_prio.keys()):
                grp_dict = groups_by_prio[prio]
                grp_keys = sorted(grp_dict.keys())
                rng.shuffle(grp_keys)
                for gk in grp_keys:
                    grp_items = grp_dict[gk]
                    if len(chosen) + len(grp_items) <= max_per_class:
                        chosen.extend(grp_items)
                    elif len(chosen) < max_per_class and len(grp_items) == 1:
                        chosen.extend(grp_items)
        else:
            chosen = items

        # Split 70/15/15 estratificado por fuente y agrupado por group_id (cero fuga por paciente)
        by_src_groups: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        for it in chosen:
            by_src_groups[it["source"]][it["group_id"]].append(it)

        train_items, val_items, test_items = [], [], []
        for src_k in sorted(by_src_groups.keys()):
            g_dict = by_src_groups[src_k]
            g_keys = sorted(g_dict.keys())
            rng.shuffle(g_keys)

            total_src = sum(len(g_dict[k]) for k in g_keys)
            target_val = max(1, round(total_src * 0.15)) if len(g_keys) >= 4 else 0
            target_test = max(1, round(total_src * 0.15)) if len(g_keys) >= 4 else 0

            val_buf, test_buf, train_buf = [], [], []
            for gk in g_keys:
                grp = g_dict[gk]
                if len(val_buf) < target_val:
                    val_buf.extend(grp)
                elif len(test_buf) < target_test:
                    test_buf.extend(grp)
                else:
                    train_buf.extend(grp)
            train_items.extend(train_buf)
            val_items.extend(val_buf)
            test_items.extend(test_buf)

        for split_name, split_list in [("train", train_items), ("val", val_items), ("test", test_items)]:
            for rec in split_list:
                out_name = f"{rec['uid']}.jpg"
                out_path = UNIFIED_DIR / split_name / cls / out_name
                shutil.copy2(rec["staging_path"], out_path)
                rec_copy = dict(rec)
                rec_copy["split"] = split_name
                rec_copy["rel_path"] = f"dataset_unified/{split_name}/{cls}/{out_name}"
                selected_records.append(rec_copy)

    full_csv = META_DIR / "manifest_full.csv"
    fieldnames = [
        "uid", "group_id", "class", "source", "archive", "orig_entry",
        "width", "height", "file_sha256", "pixel_sha256", "dhash",
    ]
    with open(full_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in raw_candidates:
            writer.writerow({k: r[k] for k in fieldnames})

    bal_csv = META_DIR / "manifest_balanced.csv"
    bal_fields = fieldnames + ["split", "rel_path"]
    with open(bal_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bal_fields)
        writer.writeheader()
        for r in selected_records:
            writer.writerow({k: r[k] for k in bal_fields})

    # Verificación estricta de 0 solapamiento entre splits por hash y por group_id
    split_hashes = defaultdict(set)
    split_groups = defaultdict(set)
    for r in selected_records:
        split_hashes[r["split"]].add(r["pixel_sha256"])
        split_groups[r["split"]].add(r["group_id"])

    overlap_tv = len(split_hashes["train"] & split_hashes["val"])
    overlap_tt = len(split_hashes["train"] & split_hashes["test"])
    overlap_vt = len(split_hashes["val"] & split_hashes["test"])

    grp_overlap_tv = len(split_groups["train"] & split_groups["val"])
    grp_overlap_tt = len(split_groups["train"] & split_groups["test"])
    grp_overlap_vt = len(split_groups["val"] & split_groups["test"])

    shutil.rmtree(staging_dir, ignore_errors=True)

    seg_records = extract_segmentation_pairs()

    dedup_report = {
        "seed": SEED,
        "max_per_class": max_per_class,
        "excluded_archives": {
            "b1b3k47_wound_cls.zip": "Excluido tras auditoría forense: contiene copias sintéticas Augmentor (*_original_*) y 512/516 duplicados SHA-256 de uwm_cls_original.zip (mismos pacientes AZH).",
            "mlaradji_wound_care.zip (google-images-medetec-*)": "Excluido tras inspección visual: contiene imágenes web no clínicas (diagramas anatómicos, infografías de texto y objetos ajenos); se conserva únicamente medetec-dataset/.",
        },
        "total_scanned": dedup_stats["total_scanned"],
        "corrupt_or_too_small": dedup_stats["corrupt_or_too_small"],
        "azh_same_case_skipped": dedup_stats["azh_same_case_skipped"],
        "exact_byte_duplicates": dedup_stats["exact_byte_duplicates"],
        "exact_pixel_duplicates": dedup_stats["exact_pixel_duplicates"],
        "perceptual_near_duplicates": dedup_stats["perceptual_near_duplicates"],
        "flipped_duplicates": dedup_stats["flipped_duplicates"],
        "accepted_unique_pool": dedup_stats["accepted_unique_pool"],
        "balanced_total": len(selected_records),
        "split_hash_overlap": {
            "train_val": overlap_tv,
            "train_test": overlap_tt,
            "val_test": overlap_vt,
        },
        "split_group_overlap": {
            "train_val": grp_overlap_tv,
            "train_test": grp_overlap_tt,
            "val_test": grp_overlap_vt,
        },
        "class_split_counts": {
            cls: dict(Counter(r["split"] for r in selected_records if r["class"] == cls))
            for cls in CANONICAL_CLASSES
        },
        "class_source_counts": {
            cls: dict(Counter(r["source"] for r in selected_records if r["class"] == cls))
            for cls in CANONICAL_CLASSES
        },
        "segmentation_pairs_count": len(seg_records),
        "segmentation_sources": dict(Counter(r["source"] for r in seg_records)),
        "by_archive": dict(dedup_stats["by_archive"]),
    }

    with open(META_DIR / "deduplication_report.json", "w", encoding="utf-8") as f:
        json.dump(dedup_report, f, indent=2, ensure_ascii=False)

    print("\n=== [FASE 4] Generando evidencias visuales en figures/ ===")
    generate_visual_evidence(selected_records, seg_records)

    print("\n=== RESUMEN FINAL DEL DATASET INTEGRADO ===")
    print(f"Total escaneado: {dedup_stats['total_scanned']} | Pool único desduplicado: {dedup_stats['accepted_unique_pool']}")
    print(
        f"Duplicados removidos -> bytes={dedup_stats['exact_byte_duplicates']}, "
        f"píxeles={dedup_stats['exact_pixel_duplicates']}, "
        f"perceptual={dedup_stats['perceptual_near_duplicates']}, "
        f"espejo={dedup_stats['flipped_duplicates']}, "
        f"azh_mismo_caso={dedup_stats['azh_same_case_skipped']}"
    )
    print(f"Solapamiento hashes -> train∩val={overlap_tv}, train∩test={overlap_tt}, val∩test={overlap_vt}")
    print(f"Solapamiento grupos -> train∩val={grp_overlap_tv}, train∩test={grp_overlap_tt}, val∩test={grp_overlap_vt}")
    print("\nDistribución en dataset_unified (train / val / test):")
    grand_total = 0
    for cls in CANONICAL_CLASSES:
        c = dedup_report["class_split_counts"][cls]
        tot = sum(c.values())
        grand_total += tot
        srcs = dedup_report["class_source_counts"][cls]
        print(f"  {cls:22s}: total={tot:4d} (train={c.get('train',0):3d}, val={c.get('val',0):3d}, test={c.get('test',0):3d}) | {srcs}")
    print(f"  {'TOTAL CLASIFICACIÓN':22s}: {grand_total:4d}")
    print(f"  {'TOTAL SEGMENTACIÓN':22s}: {len(seg_records):4d} pares imagen+máscara")


def normalize_mendeley_seg_key(entry: str) -> str:
    stem = Path(entry).stem.lower()
    for prefix in ("wound_main-", "wound_mask-", "wound_main_", "wound_mask_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def extract_segmentation_pairs() -> List[Dict]:
    """
    Integra pares imagen + máscara binaria de segmentación de área de úlcera desde:
      1) FUSeg MICCAI 2021 (uwm_segmentation_repo.zip -> Foot Ulcer Segmentation Challenge)
      2) Mendeley Lower Limb and Feet Wound Image Dataset v3 (wound_main + wound_mask)
    """
    print("\n=== [SEGMENTACIÓN] Extrayendo pares imagen + máscara binaria ===")
    seg_records: List[Dict] = []
    seg_csv = META_DIR / "manifest_segmentation.csv"

    # 1. FUSeg Challenge (train + validation)
    uwm_seg_zip = RAW_DIR / "uwm_segmentation_repo.zip"
    if uwm_seg_zip.exists():
        with zipfile.ZipFile(uwm_seg_zip, "r") as zf:
            names = zf.namelist()
            for split_dir in ["train", "validation"]:
                img_prefix = f"data/Foot Ulcer Segmentation Challenge/{split_dir}/images/"
                lbl_prefix = f"data/Foot Ulcer Segmentation Challenge/{split_dir}/labels/"
                imgs, lbls = {}, {}
                for n in names:
                    if not is_valid_image_entry(n):
                        continue
                    if img_prefix in n:
                        imgs[Path(n).stem] = n
                    elif lbl_prefix in n:
                        lbls[Path(n).stem] = n
                common = sorted(set(imgs.keys()) & set(lbls.keys()))
                for stem in common:
                    try:
                        img_raw = zf.read(imgs[stem])
                        msk_raw = zf.read(lbls[stem])
                        with Image.open(io.BytesIO(img_raw)) as im, Image.open(io.BytesIO(msk_raw)) as mk:
                            rgb = im.convert("RGB")
                            mk_gray = mk.convert("L")
                            if rgb.size != mk_gray.size:
                                mk_gray = mk_gray.resize(rgb.size, Image.Resampling.NEAREST)
                            mk_arr = (np.asarray(mk_gray) > 127).astype(np.uint8) * 255
                            if mk_arr.sum() == 0:
                                continue
                            uid = f"fuseg_{split_dir}_{stem}"
                            rgb.save(SEG_DIR / "images" / f"{uid}.png", format="PNG")
                            Image.fromarray(mk_arr).save(SEG_DIR / "masks" / f"{uid}.png", format="PNG")
                            area_ratio = float((mk_arr > 0).mean())
                            seg_records.append(
                                {
                                    "uid": uid,
                                    "source": "FUSeg_UWM",
                                    "split": "train" if split_dir == "train" else "val",
                                    "width": rgb.size[0],
                                    "height": rgb.size[1],
                                    "wound_area_px": int((mk_arr > 0).sum()),
                                    "wound_area_pct": round(area_ratio * 100, 3),
                                    "image_rel_path": f"dataset_segmentation/images/{uid}.png",
                                    "mask_rel_path": f"dataset_segmentation/masks/{uid}.png",
                                }
                            )
                    except Exception:
                        continue

    # 2. Mendeley wound_main + wound_mask (emparejamiento por sufijo numérico XXXX)
    mend_zip = RAW_DIR / "mendeley_hsj38fwnvr.zip"
    if mend_zip.exists():
        with zipfile.ZipFile(mend_zip, "r") as zf:
            names = zf.namelist()
            mains, masks = {}, {}
            for n in names:
                if not is_valid_image_entry(n):
                    continue
                nl = n.lower()
                if "/wound_main/" in nl or nl.startswith("wound_main/"):
                    mains[normalize_mendeley_seg_key(n)] = n
                elif "/wound_mask/" in nl or nl.startswith("wound_mask/") or "/wound_masked/" in nl or nl.startswith("wound_masked/"):
                    masks[normalize_mendeley_seg_key(n)] = n
            common = sorted(set(mains.keys()) & set(masks.keys()))
            rng = random.Random(SEED)
            if len(common) > 600:
                common = sorted(rng.sample(common, 600))
            for idx, key in enumerate(common):
                try:
                    img_raw = zf.read(mains[key])
                    msk_raw = zf.read(masks[key])
                    with Image.open(io.BytesIO(img_raw)) as im, Image.open(io.BytesIO(msk_raw)) as mk:
                        rgb = im.convert("RGB")
                        mk_gray = mk.convert("L")
                        if rgb.size != mk_gray.size:
                            mk_gray = mk_gray.resize(rgb.size, Image.Resampling.NEAREST)
                        mk_arr = (np.asarray(mk_gray) > 127).astype(np.uint8) * 255
                        area_pct = float((mk_arr > 0).mean())
                        if area_pct <= 0.001 or area_pct >= 0.98:
                            continue
                        safe_key = "".join(c if c.isalnum() or c in "_-" else "_" for c in key)
                        uid = f"mendeley_{idx:04d}_{safe_key}"
                        rgb.save(SEG_DIR / "images" / f"{uid}.png", format="PNG")
                        Image.fromarray(mk_arr).save(SEG_DIR / "masks" / f"{uid}.png", format="PNG")
                        split_name = "train" if idx % 10 < 7 else ("val" if idx % 10 < 8 else "test")
                        seg_records.append(
                            {
                                "uid": uid,
                                "source": "Mendeley_hsj38fwnvr",
                                "split": split_name,
                                "width": rgb.size[0],
                                "height": rgb.size[1],
                                "wound_area_px": int((mk_arr > 0).sum()),
                                "wound_area_pct": round(area_pct * 100, 3),
                                "image_rel_path": f"dataset_segmentation/images/{uid}.png",
                                "mask_rel_path": f"dataset_segmentation/masks/{uid}.png",
                            }
                        )
                except Exception:
                    continue

    with open(seg_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["uid", "source", "split", "width", "height", "wound_area_px", "wound_area_pct", "image_rel_path", "mask_rel_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in seg_records:
            writer.writerow(r)

    print(f"  Pares de segmentación guardados: {len(seg_records)} ({dict(Counter(r['source'] for r in seg_records))})")
    return seg_records


def generate_visual_evidence(selected_records: List[Dict], seg_records: List[Dict]) -> None:
    rng = random.Random(SEED)

    # 1. Mosaico 5x5 de las 5 clases clínicas
    fig, axes = plt.subplots(5, 5, figsize=(16, 16))
    for r_idx, cls in enumerate(CANONICAL_CLASSES):
        cls_recs = [r for r in selected_records if r["class"] == cls]
        # Seleccionar intentando mostrar variedad de fuentes
        by_s = defaultdict(list)
        for cr in cls_recs:
            by_s[cr["source"]].append(cr)
        picks = []
        src_keys = sorted(by_s.keys())
        while len(picks) < 5 and any(by_s.values()):
            for sk in src_keys:
                if by_s[sk] and len(picks) < 5:
                    pick = rng.choice(by_s[sk])
                    by_s[sk].remove(pick)
                    picks.append(pick)
        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            ax.axis("off")
            if c_idx < len(picks):
                rec = picks[c_idx]
                img_path = PROJECT_ROOT / rec["rel_path"]
                with Image.open(img_path) as im:
                    ax.imshow(im.resize((256, 256), Image.Resampling.BILINEAR))
                title = f"{rec['source']}\n({rec['width']}x{rec['height']})"
                if c_idx == 0:
                    title = f"[{CLASS_DISPLAY_ES[cls]}]\n{title}"
                ax.set_title(title, fontsize=9, fontweight="bold" if c_idx == 0 else "normal")
    fig.suptitle(
        "Muestra Curada y Desduplicada — 5 Clases Clínicas (Opción B: Heridas Crónicas)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "sample_grid_5_classes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Gráfico de distribución por clase, fuente y split
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    sources_present = sorted({r["source"] for r in selected_records})
    colors_src = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
    x = np.arange(len(CANONICAL_CLASSES))
    bottom = np.zeros(len(CANONICAL_CLASSES))

    for idx, src in enumerate(sources_present):
        vals = np.array([sum(1 for r in selected_records if r["class"] == c and r["source"] == src) for c in CANONICAL_CLASSES])
        ax1.bar(x, vals, bottom=bottom, label=src, color=colors_src[idx % len(colors_src)])
        bottom += vals

    for i, tot in enumerate(bottom):
        ax1.text(i, tot + 8, f"{int(tot)}", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([CLASS_DISPLAY_ES[c] for c in CANONICAL_CLASSES], rotation=18, ha="right")
    ax1.set_ylabel("Número de imágenes únicas")
    ax1.set_title("Distribución por Clase y Fuente Académica", fontweight="bold")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    split_colors = {"train": "#2b5c8f", "val": "#f2a900", "test": "#e15759"}
    bottom_s = np.zeros(len(CANONICAL_CLASSES))
    for sp in ["train", "val", "test"]:
        vals = np.array([sum(1 for r in selected_records if r["class"] == c and r["split"] == sp) for c in CANONICAL_CLASSES])
        ax2.bar(x, vals, bottom=bottom_s, label=f"{sp.upper()}", color=split_colors[sp])
        for i, v in enumerate(vals):
            if v > 20:
                ax2.text(i, bottom_s[i] + v / 2, f"{int(v)}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        bottom_s += vals

    ax2.set_xticks(x)
    ax2.set_xticklabels([CLASS_DISPLAY_ES[c] for c in CANONICAL_CLASSES], rotation=18, ha="right")
    ax2.set_ylabel("Número de imágenes únicas")
    ax2.set_title("Partición Estratificada por Paciente (70% Train / 15% Val / 15% Test)", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Muestras de segmentación de área de úlcera
    if seg_records:
        sample_seg = []
        fuseg_s = [r for r in seg_records if r["source"] == "FUSeg_UWM"]
        mend_s = [r for r in seg_records if r["source"] == "Mendeley_hsj38fwnvr"]
        if fuseg_s:
            sample_seg.extend(rng.sample(fuseg_s, min(2, len(fuseg_s))))
        if mend_s:
            sample_seg.extend(rng.sample(mend_s, min(2, len(mend_s))))

        n_rows = len(sample_seg)
        fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.6 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for idx, rec in enumerate(sample_seg):
            img_rgb = np.array(Image.open(PROJECT_ROOT / rec["image_rel_path"]).convert("RGB"))
            msk = np.array(Image.open(PROJECT_ROOT / rec["mask_rel_path"]).convert("L"))
            overlay = img_rgb.copy()
            contours, _ = cv2.findContours((msk > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 100), 3)

            axes[idx, 0].imshow(img_rgb)
            axes[idx, 0].set_title(f"{rec['source']} — Imagen Clínica", fontsize=9)
            axes[idx, 1].imshow(msk, cmap="gray")
            axes[idx, 1].set_title("Máscara Binaria Ground-Truth", fontsize=9)
            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title(
                f"Área Úlcera: {rec['wound_area_px']:,} px ({rec['wound_area_pct']:.2f}%)",
                fontsize=9,
                fontweight="bold",
            )
            for c in range(3):
                axes[idx, c].axis("off")

        fig.suptitle(
            "Módulo de Seguimiento Cuantitativo — Segmentación y Cálculo de Área de Úlcera",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        fig.savefig(FIG_DIR / "segmentation_samples.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga e integración del dataset de heridas crónicas.")
    parser.add_argument("--skip-download", action="store_true", help="Omite la descarga si los ZIPs ya existen.")
    parser.add_argument("--inspect-only", action="store_true", help="Solo descarga e inspecciona las carpetas internas.")
    parser.add_argument("--max-per-class", type=int, default=500, help="Tope de imágenes balanceadas por clase.")
    args = parser.parse_args()

    ensure_dirs()
    if not args.skip_download:
        download_all()

    if args.inspect_only:
        inspect_archives()
        return

    build_datasets(max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
