# skin-wound-tracker

Computer vision application for skin wound classification and longitudinal monitoring through area, tissue coloration, inflammation, and exudate analysis.

**Proyecto Final — Visión Computacional con Deep Learning (UAO, 2026-1)**  
**Enfoque:** Heridas Crónicas / Clínicas (5 clases de clasificación + segmentación binaria para cuantificación de área), trabajando exclusivamente con **imágenes clínicas reales**.

---

## 1. Arquitectura y Pipeline

Consulta el diseño completo bloque por bloque (`01 DATOS` → `05 INTERFAZ / USO`), anatomía de bloques y las 5 preguntas encadenadas en [`docs/PIPELINE.md`](docs/PIPELINE.md).

---

## 2. Fuentes de Datos Clínicas (Solo Imágenes RGB)

Todas las fuentes descargadas corresponden a fotografía clínica hospitalaria real (excluyendo colecciones con copias sintéticas `Augmentor` como `b1b3k47` y capturas web no clínicas `google-images-medetec-*`):

| Fuente académica | Repositorio / Origen | Uso en el proyecto |
| :--- | :--- | :--- |
| **AZH / UWM** | [`VincenzoPalma/WoundTypeClassifier`](https://github.com/VincenzoPalma/WoundTypeClassifier) + [`uwm-bigdata/wound_classification`](https://github.com/uwm-bigdata/wound_classification) | 5 clases (`D, P, V, S, N`; se descarta `BG`) |
| **Medetec Wound Database** | [`mlaradji/deep-learning-for-wound-care`](https://github.com/mlaradji/deep-learning-for-wound-care) *(solo `data/medetec-dataset/`)* | 4 clases de heridas (`pressure`, `venous`, `diabetic`, `surgical`) |
| **SurgWound (OSWUMC)** | [`xuxuxuxuxu/SurgWound`](https://huggingface.co/datasets/xuxuxuxuxu/SurgWound) *(solo imágenes, sin textos)* | `surgical_wound` (686 imágenes únicas) |
| **FUSeg (MICCAI 2021)** | [`uwm-bigdata/wound-segmentation`](https://github.com/uwm-bigdata/wound-segmentation) | `diabetic_foot_ulcer` + pares imagen/máscara de segmentación |
| **Mendeley v3 (`hsj38fwnvr`)** | [Mendeley Data `hsj38fwnvr`](https://data.mendeley.com/datasets/hsj38fwnvr/3) | `normal_skin` (`Nomal/`) + pares imagen/máscara de segmentación |
| **PIID** *(candidato adicional)* | [`FU-MedicalAI/PIID`](https://github.com/FU-MedicalAI/PIID) | +1,091 imágenes RGB de `pressure_ulcer` (Estadios 1–4) |

---

## 3. Estado Actual de Auditoría y Manifiestos

Los datos pesados (`raw_downloads/`, `dataset_unified/`, `dataset_segmentation/`) están excluidos en `.gitignore` y se reproducen con `scripts/download_and_integrate.py`. En este repositorio se incluyen los manifiestos auditados y evidencias visuales:

- **Pool único desduplicado (`metadata/manifest_full.csv`):** `5,695` imágenes clínicas únicas (`diabetic_foot_ulcer=1,250`, `pressure_ulcer=308`, `venous_ulcer=384`, `surgical_wound=910`, `normal_skin=2,843`).
- **Muestra balanceada de referencia (`metadata/manifest_balanced.csv`):** `2,192` imágenes (`train=1,528`, `val=332`, `test=332`) con **0 solapamiento de hashes y 0 fuga por paciente** ([`metadata/deduplication_report.json`](metadata/deduplication_report.json)).
- **Segmentación (`metadata/manifest_segmentation.csv`):** `1,586` pares imagen + máscara binaria (`FUSeg=986`, `Mendeley=600`).

---

## 4. Estructura de la Rama `develop`

```text
skin-wound-tracker/
├── docs/
│   └── PIPELINE.md               # Diagrama de bloques (01–05), anatomía y 5 preguntas encadenadas
├── scripts/
│   ├── download_and_integrate.py # Descarga reproducible, auditoría SHA-256/dHash y splits por paciente
│   └── wound_dataset.py          # Datasets y DataLoaders en PyTorch (clasificación + segmentación)
├── metadata/
│   ├── deduplication_report.json # Auditoría de desduplicación y cero data leakage
│   ├── manifest_full.csv         # Pool completo desduplicado (5,695 imágenes)
│   ├── manifest_balanced.csv     # Subconjunto balanceado de referencia (2,192 imágenes)
│   └── manifest_segmentation.csv # 1,586 pares imagen + máscara binaria
├── figures/
│   ├── sample_grid_5_classes.png # Cuadrícula 5x5 verificada de fotografía clínica real
│   ├── class_distribution.png    # Distribución por clase, fuente y partición
│   └── segmentation_samples.png  # Ejemplos de máscaras binarias y área calculada
└── README.md
```

---

## 5. Reproducción Rápida

```bash
# 1. Descargar fuentes, auditar duplicados y construir datasets localmente
python scripts/download_and_integrate.py --max-per-class 500

# 2. Verificar PyTorch Datasets y DataLoaders
python scripts/wound_dataset.py
```
