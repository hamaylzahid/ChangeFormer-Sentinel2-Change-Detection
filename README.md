<br>
<h1 align="center">CHANGEFORMER: SENTINEL-2 CHANGE DETECTION</h1>
<br>

<p align="center">
A transformer-based deep learning framework for multi-temporal Sentinel-2 imagery to detect vegetation, urban, and environmental changes.<br>
Leverages patch embeddings, NDVI/NDBI indices, and a patch-based Transformer architecture to produce pixel-wise change maps with high accuracy and efficiency.
</p>

<br>
<p align="center">
  
<!-- Badges -->
<img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python">
<img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="PyTorch">
<img src="https://img.shields.io/badge/License-MIT-green" alt="License">
<img src="https://img.shields.io/badge/Contact-@hamaylzahid-lightgrey" alt="Contact">
<img src="https://img.shields.io/badge/Status-Active-success">
</p>

<br>
<h2 align="center">Table of Contents</h2>
<br>

<p align="center">
  
- [Introduction](#introduction)<br>
- [Project Overview](#project-overview)<br>
- [Data Acquisition & Preprocessing](#data-acquisition--preprocessing)<br>
- [Model Architecture](#model-architecture)<br>
- [Training & Evaluation](#training--evaluation)<br>
- [Results & Visualizations](#results--visualizations)<br>
- [Performance Metrics](#performance-metrics)<br>
- [Applications & Impact](#applications--impact)<br>
- [Conclusion](#conclusion)<br>
- [Contact & License](#contact--license)
</p>

<br>
<h2 align="center" id="introduction">Introduction</h2>
<br>

<p align="center">
This repository implements <strong>ChangeFormer</strong>, a deep learning model for automated change detection in multi-temporal Sentinel-2 images.<br>
Key highlights:
</p>

<p align="center">
  
- Patch-based Transformer architecture<br>
- Spectral indices (NDVI & NDBI) for vegetation and urban change detection<br>
- Cloud filtering and augmentation for robust model training<br>
- Generates accurate pixel-wise change maps suitable for environmental monitoring and urban planning
</p>

<br>
<h2 align="center" id="project-overview">Project Overview</h2>
<br>

<p align="center">
  
- Detects temporal changes between Sentinel-2 images using Transformer-based embeddings<br>
- Handles multispectral data including RGB, NIR, and SWIR bands<br>
- Patch extraction and augmentation ensure a diverse and robust training set<br>
- Outputs include visual change maps and quantitative evaluation metrics
</p>

<br>
<h2 align="center" id="data-acquisition--preprocessing">Data Acquisition & Preprocessing</h2>
<br>

<p align="center">
  
- Access Sentinel-2 imagery via <strong>Google Earth Engine</strong> with &lt; 20% cloud coverage<br>
- Selected bands: B2, B3, B4, B8, B11<br>
- Computed indices: <strong>NDVI</strong> (vegetation) and <strong>NDBI</strong> (built-up areas)<br>
- Converted EE images to NumPy arrays and segmented into overlapping patches<br>
- Data augmentation: flipping, rotation, brightness/contrast adjustment, and translation to increase variability
</p>

<br>
<h2 align="center" id="model-architecture">Model Architecture</h2>
<br>

<p align="center">
  
- <strong>Patch Embedding:</strong> Converts image patches into feature vectors<br>
- <strong>Transformer Blocks:</strong> Capture temporal differences between t1 and t2<br>
- <strong>Decoder:</strong> Reconstructs pixel-wise change maps<br>
- <strong>Loss:</strong> BCE + Dice loss for stable and accurate training
</p>

<br>
<h2 align="center" id="training--evaluation">Training & Evaluation</h2>
<br>

<p align="center">
  
- Optimizer: <strong>AdamW</strong> with learning rate 1e-4<br>
- Batch-wise loading of augmented patches<br>
- Training on NVIDIA GPUs or compatible Colab runtime<br>
- Evaluation metrics: IoU, Precision, Recall, F1-score<br>
- Training shows stable convergence with smooth loss curves
</p>

<br>
<h2 align="center" id="results--visualizations">Results & Visualizations</h2>
<br>

<h3 align="center">Training Loss Curve</h3>
<br>
<p align="center">
<img src="outputs/Trainingloss.png" alt="Training Loss" width="600px">
</p>
<p align="center">Smooth convergence of the training loss over 20 epochs indicates robust model optimization.</p>

<h3 align="center">NDVI Change Distribution</h3>
<br>
<p align="center">
<img src="outputs/NVDI_chnage_destribution.png" alt="NDVI Distribution" width="600px">
</p>
<p align="center">Distribution shows areas with significant vegetation changes between two time points.</p>

<h3 align="center">Before & After Comparison</h3>
<br>
<p align="center">
<img src="outputs/comparison.png" alt="Comparison Map" width="600px">
</p>
<p align="center">Pixel-wise visualization clearly demonstrates vegetation growth and urban expansion.</p>

<br>
<h2 align="center" id="performance-metrics">Performance Metrics</h2>
<br>

<p align="center">
The following table summarizes the model performance for ChangeFormer on Sentinel-2 change detection:
</p>

<br>

<table align="center" border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse;">
  <thead>
    <tr style="background-color:#4CAF50; color:white;">
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>IoU</td>
      <td>0.5338</td>
    </tr>
    <tr>
      <td>Precision</td>
      <td>0.6448</td>
    </tr>
    <tr>
      <td>Recall</td>
      <td>0.7563</td>
    </tr>
    <tr>
      <td>F1-score</td>
      <td>0.6961</td>
    </tr>
  </tbody>
</table>
<p align="center">Metrics indicate strong detection capability with minimal false positives.</p>

<br>
<h2 align="center" id="applications--impact">Applications & Impact</h2>
<br>

<p align="center">
  
- Environmental monitoring: deforestation, vegetation growth, degradation<br>
- Urban planning: infrastructure expansion and land-use change<br>
- Disaster assessment: floods, wildfires, or environmental hazards<br>
- Scientific research: long-term temporal analysis of ecological and urban changes
</p>

<br>
<h2 align="center" id="conclusion">Conclusion</h2>
<br>

<p align="center">
ChangeFormer demonstrates that Transformer-based architectures are effective for multispectral change detection.<br>
Patch-based embeddings combined with NDVI/NDBI indices provide automated, accurate detection supporting various real-world applications.
</p>

<br>
<h2 align="center">Contact & Contribution</h2>
<br>

<p align="center">
Have feedback, want to collaborate, or want to extend this project?<br>
<strong>Let’s connect and enhance multispectral change detection and environmental monitoring together.</strong>
</p>

<p align="center">
Email: <a href="mailto:hamaylzahid@gmail.com">hamaylzahid@gmail.com</a> &nbsp; | &nbsp;
LinkedIn: <a href="https://www.linkedin.com/in/hamaylzahid">Profile</a> &nbsp; | &nbsp;
GitHub Repo: <a href="https://github.com/hamaylzahid/ChangeFormer-Sentinel2-Change-Detection">Repository</a>
</p>

<p align="center">
<a href="https://github.com/hamaylzahid/ChangeFormer-Sentinel2-Change-Detection/stargazers">
<img src="https://img.shields.io/badge/Star%20This%20Project-Give%20a%20Star-yellow?style=for-the-badge&logo=github" alt="Star Badge">
</a>
<a href="https://github.com/hamaylzahid/ChangeFormer-Sentinel2-Change-Detection/pulls">
<img src="https://img.shields.io/badge/Contribute-Pull%20Requests%20Welcome-2ea44f?style=for-the-badge&logo=github" alt="PR Badge">
</a>
</p>

<p align="center">
Found this project useful? Give it a star.<br>
Want to improve it? Submit a pull request and join the development.<br>
</p>

<br>
<h2 align="center">License</h2>
<br>

<p align="center">
This project is licensed under the <strong>MIT License</strong> and is open for use, modification, and distribution.
</p>

<p align="center">
<strong>Developed with deep learning, Sentinel-2 imagery, and environmental monitoring principles in mind.</strong>
</p>

<p align="center">
<a href="https://github.com/hamaylzahid">
<img src="https://img.shields.io/badge/GitHub-%40hamaylzahid-181717?style=flat-square&logo=github" alt="GitHub">
</a>
&nbsp;•&nbsp;
<a href="mailto:hamaylzahid@gmail.com">
<img src="https://img.shields.io/badge/Email-Contact%20Me-red?style=flat-square&logo=gmail&logoColor=white" alt="Email">
</a>
&nbsp;•&nbsp;
<a href="https://github.com/hamaylzahid/ChangeFormer-Sentinel2-Change-Detection">
<img src="https://img.shields.io/badge/Repo-Link-blueviolet?style=flat-square&logo=github" alt="Repo">
</a>
<br>
<a href="https://github.com/hamaylzahid/ChangeFormer-Sentinel2-Change-Detection/fork">
<img src="https://img.shields.io/badge/Fork%20This%20Project-Contribute-2ea44f?style=flat-square&logo=github" alt="Fork">
</a>
</p>

<p align="center">
<sub><i>Designed for multispectral change detection, environmental monitoring, and automated change mapping.</i></sub>
</p>
