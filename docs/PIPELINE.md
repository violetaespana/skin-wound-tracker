# Pipeline del Sistema — `skin-wound-tracker`

Diseño bloque por bloque siguiendo el estándar UAO (*De la imagen a la acción* y *Anatomía de un bloque*: Nombre, Función, Entrada, Salida).  
**Regla de alcance actual:** el sistema y los datasets operan exclusivamente sobre **imágenes clínicas RGB** (sin textos ni formularios de metadatos externos).

---

## 1. Diagrama del Pipeline

```mermaid
flowchart LR
    subgraph OFFLINE["FASE OFFLINE: DATOS Y MODELOS"]
        direction LR
        P1["00A · RECOLECCIÓN\nfetch_raw_datasets"] -->|"imágenes crudas\n(.jpg / .png)"| P2["00B · PREPARACIÓN\naudit_and_split"]
        P2 -->|"splits train/val/test\n+ manifiestos CSV"| P3["00C · ENTRENAMIENTO\ntrain_and_evaluate"]
    end

    subgraph ONLINE["SISTEMA FUNCIONAL: DE LA IMAGEN A LA ACCIÓN (01–05)"]
        direction LR
        B1["01 · DATOS\ningest_clinical_image"] -->|"imagen RGB\nvalidada"| B2["02 · PREPROCESO\npreprocess_dual_inputs"]
        B2 -->|"tensor x_cls\n(1, 3, 224, 224)"| B3A["03A · INFERENCIA CLS\ninfer_wound_classifier"]
        B2 -->|"tensor x_seg\n(1, 3, 256, 256)"| B3B["03B · INFERENCIA SEG\ninfer_wound_segmenter"]
        B3A -->|"señal condicional\n(clase != normal_skin)"| B3B
        B3A -->|"probs (5,) + clase +\nmapa Grad-CAM"| B4["04 · POSPROCESO\npostprocess_clinical_metrics"]
        B3B -->|"mask_prob\n(1, 1, 256, 256)"| B4
        DB[("DB · HISTORIAL\npatient_history_store")] <-->|"visita previa /\nguarda medición"| B4
        B4 -->|"JSON + indicadores +\noverlays visuales"| B5["05 · INTERFAZ / USO\nclinical_dashboard_action"]
        DB -->|"serie temporal"| B5
    end

    P3 -->|"best_classifier.pt"| B3A
    P3 -->|"best_unet.pt"| B3B
```

---

## 2. Anatomía de Bloques (Nombre · Función · Entrada · Salida)

| Bloque (Diagrama / Código) | Función (en una frase) | Entrada | Salida |
| :--- | :--- | :--- | :--- |
| **`01 · DATOS`**<br>`ingest_clinical_image()` | Carga y valida la fotografía clínica RGB antes de procesarla. | Archivo `.jpg`/`.png` RGB desde cámara o repositorio. | `PIL.Image` RGB validada ($\ge 64\times64$, EXIF corregido) + `orig_size (H, W)`. |
| **`02 · PREPROCESO`**<br>`preprocess_dual_inputs()` | Redimensiona y normaliza la imagen a las dimensiones que espera cada modelo. | `PIL.Image` RGB desde `01 · DATOS`. | `x_cls` `FloatTensor (1, 3, 224, 224)` y `x_seg` `FloatTensor (1, 3, 256, 256)` (normalización ImageNet). |
| **`03A · INFERENCIA CLS`**<br>`infer_wound_classifier()` | Clasifica la imagen en una de las 5 clases clínicas y genera el mapa de atención Grad-CAM. | `x_cls` (`02`) + pesos `best_classifier.pt`. | `pred_class`, `confidence`, `probs (5,)`, `gradcam_map (H, W)` y bandera `has_wound = (pred_class != "normal_skin")`. |
| **`03B · INFERENCIA SEG`**<br>`infer_wound_segmenter()` | Segmenta píxel a píxel la región de la herida solo cuando existe lesión activa. | `x_seg` (`02`) + señal `has_wound == True` (`03A`) + pesos `best_unet.pt`. | `mask_prob` float32 `(256, 256)` en `[0, 1]` (o `None` si `normal_skin`). |
| **`04 · POSPROCESO`**<br>`postprocess_clinical_metrics()` | Convierte predicciones y máscara en métricas de área, variación longitudinal y alertas clínicas. | Salidas de `03A`, `03B` y registro previo de `DB · HISTORIAL`. | `clinical_report.json` (`wound_area_px`, `wound_area_pct`, `delta_area_pct`, `triage_priority`, `clinical_alert`) + overlays (`contour`, `gradcam`). |
| **`DB · HISTORIAL`**<br>`patient_history_store` | Persiste y consulta la serie temporal de mediciones de área para evaluar cicatrización. | Registro `clinical_report.json` desde `04 · POSPROCESO`. | Medición anterior hacia `04` y serie temporal hacia `05`. |
| **`05 · INTERFAZ / USO`**<br>`clinical_dashboard_action()` | Muestra el diagnóstico visual explicable, la evolución temporal y dispara alertas operativas. | `clinical_report.json`, overlays (`04`) y serie temporal (`DB`). | Panel visual + ficha clínica descargable (`JSON`) + alerta de estancamiento o revisión humana. |

---

## 3. Cinco Preguntas Encadenadas (Resumen Ejecutivo)

1. **Entrada concreta:** Fotografía clínica RGB (`.jpg`/`.png`) desde cámara o archivo; se valida integridad, resolución $\ge 64\times64$, 3 canales y orientación EXIF.
2. **Procesamiento trazable:** `01 ingest` → `02 preprocess (224×224 + 256×256)` → `03A clasificador + Grad-CAM` → `03B segmentador U-Net` *(solo si hay herida)* → `04 posproceso ↔ DB historial` → `05 interfaz`.
3. **Predicción interpretable:** JSON con `pred_class` (5 etiologías), `confidence` y `probs`, `gradcam_map`, `mask_binary`, `wound_area_px/pct`, `delta_area_pct`, `triage_priority` y `clinical_alert`.
4. **Representación visible:** Tarjeta de diagnóstico + barras de probabilidad + mapa de calor Grad-CAM + foto con contorno verde y área + gráfica temporal de evolución + semáforo de alerta y descarga JSON.
5. **Acción o valor:** Reemplaza la evaluación subjetiva por planimetría automática; emite alerta de estancamiento si el área no reduce $\ge 20\%$ entre consultas y escala a revisión humana cuando `confidence < 0.70`.
