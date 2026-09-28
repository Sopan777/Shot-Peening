# Shot Peening Surface Deformation & Coverage Prediction AI

An end-to-end Computer Vision & Deep Learning pipeline designed to predict **Surface Coverage Percentage (%)** from **SEM (Scanning Electron Microscope)** micrographs of shot-peened metal surfaces.

Optimized specifically for **Intel(R) Core(TM) Ultra 5 235 (3.40 GHz)**:
- **CPU:** Fast training using **Intel IPEX** + **BFloat16** mixed precision
- **NPU:** Ultra low-power, high-throughput deployment using **OpenVINO™**

---

## 🔬 Model Architecture: DINOv2-B + Linear Probe

- **Backbone:** Facebook's **DINOv2 (ViT-B/14)** self-supervised foundation model (trained on 142M images). Unlike ImageNet-supervised models, DINOv2 captures universal micro-textures, crater morphology, and dimple overlap without object bias.
- **Stage 1 (Linear Probe):** The 86M-parameter vision transformer backbone is **frozen** ($\approx$ 200k trainable head parameters). Trains in **~5 to 10 minutes** on your Intel CPU with **near-zero risk of overfitting** on ~1,000 images.
- **Stage 2 (Fine-Tuning - Optional):** Selectively unfreezes the top 4 transformer blocks with a low learning rate ($10^{-5}$) for maximum precision.
- **Alternative Baseline:** **EfficientNet-B3** is also fully supported via configuration.

---

## 📁 Project Structure

```
Shot-Peening/
├── data/
│   ├── images/               # Put your SEM .tif images here
│   └── labels.csv            # Mapping: filename, coverage_percentage
├── src/
│   ├── config.py             # Central hyperparameters & hardware paths
│   ├── dataset.py            # SEM TIFF loader, Albumentations & 70/15/15 split
│   ├── model.py              # DINOv2-B & EfficientNet-B3 architectures
│   ├── transforms.py         # CLAHE, rotation, blur, affine augmentations
│   ├── train.py              # Two-stage CPU/IPEX training loop & metrics
│   ├── evaluate.py           # MAE, RMSE, R² & scatter/residual plotting
│   ├── predict.py            # Single-image and batch directory inference
│   └── export_openvino.py    # OpenVINO IR exporter for Intel NPU
├── notebooks/
│   └── eda.py                # Image resolution inspection & CSV template builder
├── outputs/
│   ├── checkpoints/          # Saved model weights (.pth)
│   ├── logs/                 # TensorBoard training curves
│   ├── plots/                # High-res diagnostic figures
│   └── openvino/             # Compiled OpenVINO .xml/.bin models
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Installation

In PowerShell or your terminal:

```pwsh
# Create and activate virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

*(Optional for Intel CPU maximum speed)*:
```pwsh
pip install intel-extension-for-pytorch
```

---

### 2. Prepare Your Dataset

1. Place all your SEM `.tif` images into:
   ```
   data/images/
   ```
2. Create `data/labels.csv` with the image filenames and corresponding coverage percentages:
   ```csv
   filename,coverage_percentage
   SP Over NB 09-05-25 Stn-1---2.tif,82.4
   SP Over NB 09-05-25 Stn-2---2.tif,74.1
   SP Over NB 14-02-26 Stn-1-3.tif,91.8
   ...
   ```
   *Tip:* If you haven't created your CSV yet, run the EDA tool below. It will automatically detect your images and generate a `labels_template.csv` for you!

---

### 3. Exploratory Data Analysis & Verification

Inspect image dimensions, bit depth, and coverage label distribution:

```pwsh
python notebooks/eda.py
```

---

### 4. Train the Model (Stage 1: Linear Probe)

Train the regression head on your Intel CPU:

```pwsh
python -m src.train --model dinov2_base --epochs1 25 --batch_size 8
```

- TensorBoard monitoring:
  ```pwsh
  tensorboard --logdir outputs/logs
  ```
- Checkpoints will automatically be saved to `outputs/checkpoints/best_model.pth`.

---

### 5. Evaluate Accuracy & Generate Diagnostic Plots

Calculate test set **MAE**, **RMSE**, and **$R^2$**, and generate publication-ready plots:

```pwsh
python -m src.evaluate
```

Plots generated in `outputs/plots/`:
- `scatter_predicted_vs_actual.png` (Visualizes $R^2$ correlation against the ideal 45° line)
- `residual_distribution.png` (Validates Gaussian zero-mean error distribution)
- `test_predictions_detailed.csv` (Ranks samples by largest prediction discrepancy for error analysis)

---

### 6. Predict on New Images

#### Single SEM Image:
```pwsh
python -m src.predict --image "data/images/sample.tif"
```

Output:
```
=======================================================
Image:          sample.tif
Coverage:       89.45%
Interpretation: Nominal full coverage (High density overlapping dimples)
=======================================================
```

#### Batch Folder of Images:
```pwsh
python -m src.predict --folder "C:/path/to/new_batch/" --output_csv "batch_results.csv"
```

---

### 7. Deploy to Intel Core Ultra NPU (OpenVINO)

Export your trained PyTorch checkpoint into OpenVINO Intermediate Representation (IR) and benchmark inference latency on the NPU:

```pwsh
python -m src.export_openvino
```

This compiles `outputs/openvino/shot_peening_coverage.xml` specifically targeted for your Intel NPU accelerator.
