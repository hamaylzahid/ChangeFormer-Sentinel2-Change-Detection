<h1 align="center">ChangeFormer: Sentinel-2 Change Detection</h1>

<p align="center">
A Transformer-based framework for multi-temporal Sentinel-2 imagery, designed for pixel-wise detection of vegetation, urban, and environmental changes.
</p>

<p align="center">
<em>Research Project | Computer Vision | Remote Sensing | Change Detection</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Sentinel--2-4285F4?logo=googleearth&logoColor=white" alt="Sentinel-2">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
</p>

---

## Table of Contents

* [Introduction](#introduction)
* [Research Objective](#research-objective)
* [Project Overview](#project-overview)
* [Data Acquisition & Preprocessing](#data-acquisition--preprocessing)
* [Model Architecture](#model-architecture)
* [Training & Evaluation](#training--evaluation)
* [Results](#results)
* [Qualitative Analysis](#qualitative-analysis)
* [Applications](#applications)
* [Limitations](#limitations)
* [Future Directions](#future-directions)
* [Conclusion](#conclusion)
* [Citation](#citation)
* [Author](#author)
* [License](#license)

---

## Introduction

Multi-temporal satellite imagery provides a way to monitor how land surfaces change over time. However, automatically identifying meaningful changes from high-dimensional multispectral imagery requires models that can capture both spatial structure and temporal differences.

This project implements a Transformer-based change-detection framework for multi-temporal Sentinel-2 imagery. The pipeline combines multispectral information with derived spectral indices, including **NDVI** for vegetation-related information and **NDBI** for built-up area information, before generating pixel-wise change predictions.

The implementation is designed around patch-based processing and Transformer feature representations to model spatial and temporal differences between observations acquired at different time points.

---

## Research Objective

The project investigates the use of Transformer-based feature representations for multispectral remote-sensing change detection.

The primary objectives are to:

* Detect pixel-level changes between two Sentinel-2 observations.
* Incorporate multispectral information beyond conventional RGB imagery.
* Use NDVI and NDBI to provide additional vegetation and built-up-area information.
* Learn spatial and temporal feature relationships using Transformer blocks.
* Evaluate predictions using standard segmentation and detection metrics.

---

## Project Overview

The pipeline consists of four major stages:

```text
Sentinel-2 Image (t1) ─┐
                       ├──> Preprocessing ──> Patch Extraction
Sentinel-2 Image (t2) ─┘                         │
                                                ↓
                                  Multispectral + NDVI/NDBI
                                                │
                                                ↓
                                      Patch Embeddings
                                                │
                                                ↓
                                      Transformer Blocks
                                                │
                                                ↓
                                            Decoder
                                                │
                                                ↓
                                   Pixel-wise Change Map
```

### Key components

* Multi-temporal Sentinel-2 imagery
* Multispectral feature extraction
* NDVI and NDBI spectral indices
* Overlapping patch extraction
* Patch-based Transformer representations
* BCE + Dice loss
* Pixel-wise change-map generation
* Quantitative and qualitative evaluation

---

## Data Acquisition & Preprocessing

Sentinel-2 imagery was obtained through **Google Earth Engine** and filtered using a cloud-coverage threshold of less than 20%.

### Spectral inputs

The pipeline uses the following Sentinel-2 bands:

| Band | Description         |
| ---- | ------------------- |
| B2   | Blue                |
| B3   | Green               |
| B4   | Red                 |
| B8   | Near-infrared       |
| B11  | Short-wave infrared |

### Derived spectral indices

Two additional indices are calculated to provide domain-specific information:

**NDVI**

```text
NDVI = (NIR - Red) / (NIR + Red)
```

NDVI provides information related to vegetation condition and change.

**NDBI**

```text
NDBI = (SWIR - NIR) / (SWIR + NIR)
```

NDBI provides information related to built-up and urban areas.

### Patch processing

The preprocessed multi-temporal imagery is converted into overlapping image patches for model training and inference. Data augmentation includes:

* Horizontal and vertical flipping
* Rotation
* Brightness and contrast adjustment
* Translation

This produces additional spatial variation during training while preserving the underlying change-detection task.

---

## Model Architecture

The implementation uses a patch-based Transformer architecture for multi-temporal change detection.

### Architecture pipeline

```text
Multi-temporal Sentinel-2 Features
              │
              ↓
        Patch Extraction
              │
              ↓
       Patch Embeddings
              │
              ↓
      Transformer Blocks
              │
       Temporal / Spatial
       Feature Modeling
              │
              ↓
           Decoder
              │
              ↓
     Pixel-wise Change Map
```

### Patch Embedding

Input image patches are transformed into feature vectors that can be processed by the Transformer.

### Transformer Blocks

Transformer blocks model relationships between learned patch representations from the temporal observations, allowing the network to capture changes in spatial and spectral context.

### Decoder

The decoder reconstructs the learned representations into a pixel-wise change map corresponding to the original spatial structure.

### Loss Function

Training uses a combination of:

```text
Total Loss = BCE Loss + Dice Loss
```

BCE provides pixel-wise classification supervision, while Dice loss helps optimize overlap between predicted and reference change regions.

---

## Training & Evaluation

### Training configuration

| Setting         | Value                             |
| --------------- | --------------------------------- |
| Optimizer       | AdamW                             |
| Learning rate   | 1e-4                              |
| Loss            | BCE + Dice                        |
| Training epochs | 20                                |
| Hardware        | NVIDIA GPU / Google Colab         |
| Input           | Multi-temporal Sentinel-2 patches |

Training loss was monitored throughout the experiment to verify convergence and optimization stability.

### Evaluation metrics

The trained model was evaluated using:

* **IoU**: Intersection over Union
* **Precision**: proportion of predicted change pixels that are correct
* **Recall**: proportion of reference change pixels detected
* **F1-score**: harmonic mean of precision and recall

---

## Results

The evaluated model achieved the following results:

| Metric    |      Value |
| --------- | ---------: |
| IoU       | **0.5338** |
| Precision | **0.6448** |
| Recall    | **0.7563** |
| F1-score  | **0.6961** |

The results indicate that the model detects a substantial portion of the reference change regions, with recall reaching **75.63%** and F1-score reaching **69.61%** on the evaluated dataset.

These results are reported as the measured performance of this implementation and are not presented as a state-of-the-art comparison.

---

## Qualitative Analysis

In addition to numerical evaluation, the project generates visual change maps to inspect the spatial behavior of the model.

Representative outputs include comparisons between:

1. Sentinel-2 imagery at the first time point
2. Sentinel-2 imagery at the second time point
3. Reference change mask
4. Predicted change map

These visualizations provide qualitative insight into how the model identifies vegetation growth, urban expansion, and other surface changes.

### Training convergence

The training loss decreases progressively over the 20 training epochs, indicating stable optimization during the reported experiment.

### Spectral change analysis

NDVI difference distributions are used to examine vegetation-related changes between the two temporal observations.

---

## Applications

The resulting change maps can support several remote-sensing applications, including:

### Environmental monitoring

* Vegetation change analysis
* Deforestation monitoring
* Land degradation assessment
* Ecosystem observation

### Urban analysis

* Built-up area expansion
* Infrastructure development
* Land-use change monitoring

### Disaster assessment

The framework can potentially support change analysis following:

* Floods
* Wildfires
* Other environmental disturbances

### Scientific research

Multi-temporal change detection can provide a computational tool for studying long-term spatial and environmental dynamics from satellite imagery.

---

## Limitations

Several limitations should be considered when interpreting the reported results.

### Dataset scope

The reported metrics correspond to the evaluated Sentinel-2 dataset and experimental configuration. Additional geographic regions and independently collected temporal pairs would provide a stronger assessment of generalization.

### Limited baseline comparison

The current experiment reports the performance of the implemented model but does not yet provide a controlled comparison against multiple established change-detection baselines under identical data and training conditions.

### Ablation analysis

The individual contribution of multispectral bands, NDVI, NDBI, and the Transformer components has not yet been isolated through controlled ablation experiments.

### Spatial and temporal variability

Satellite imagery can vary substantially due to atmospheric conditions, seasonal effects, illumination, registration differences, and land-cover dynamics. Broader testing across geographic regions and acquisition periods would provide a more comprehensive evaluation.

---

## Future Directions

Future experiments include:

* Baseline comparison with CNN-based change-detection architectures
* Ablation of NDVI and NDBI features
* Multispectral-band ablation
* Transformer-component ablation
* Cross-region generalization testing
* Evaluation across different temporal intervals
* Evaluation under varying cloud and atmospheric conditions
* Higher-resolution change-map generation
* More extensive failure-case analysis

---

## Conclusion

This project explores a Transformer-based approach to pixel-wise change detection using multi-temporal Sentinel-2 imagery.

By combining multispectral information, NDVI/NDBI spectral indices, patch embeddings, and Transformer-based feature modeling, the framework provides a structured approach to detecting spatial changes across temporal satellite observations.

The reported experiment achieved **53.38% IoU, 64.48% precision, 75.63% recall, and 69.61% F1-score**, providing a quantitative baseline for further architectural and feature-level experimentation.

The next stage of the work is focused on controlled ablations, stronger baseline comparisons, and cross-region evaluation to better establish which components contribute to performance and how well the approach generalizes.

---

## Author

<p align="center">
  <strong>Hamayl Zahid</strong>
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid">
    <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://www.linkedin.com/in/hamaylzahid/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="https://hamaylzahid.github.io/hamaylzahid/">
    <img src="https://img.shields.io/badge/Portfolio-000000?logo=googlechrome&logoColor=white" alt="Portfolio">
  </a>
</p>

---

## License

This project is released under the MIT License.

Copyright © 2026 Hamayl Zahid
