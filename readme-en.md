<div align="center">
    <h1>⚡ FastPHM ⚡</h1>
</div>

<div align="center"><h3>✨ 
A fast-start, fast-executing PHM experimental framework !✨</h3></div>

<div align="center">

[![GPLv3 License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Gitee star](https://gitee.com/holdenmcgorin/FastPHM/badge/star.svg?theme=dark)](https://gitee.com/ghost-him/ZeroLaunch-rs/stargazers)
[![GitHub stars](https://img.shields.io/github/stars/holden-mcgorin/FastPHM.svg?style=social)](https://github.com/ghost-him/ZeroLaunch-rs/stargazers)

</div>

<div align="center">

[简体中文](README.md) | [English](readme-en.md)

</div>

<div align="center">
    <a href="https://gitee.com/holdenmcgorin/FastPHM" target="_blank">Gitee</a> •
    <a href="https://github.com/holden-mcgorin/FastPHM" target="_blank">GitHub</a>
</div>

### 
> A fast-start, fast-executing PHM experimental framework for rapid experiment setup and streamlined code development



## 🚀     Feature Overview
- Compatible with multiple deep learning frameworks for model development (PyTorch, TensorFlow, Pyro).
- Automatic export of experimental parameters and results (models, regularization coefficients, iteration counts, sampling counts, etc.).
- Supports various experimental subjects (bearings, turbofan engines, batteries, etc.).
- Automatic import of multiple datasets (XJTU-SY, PHM2012, C-MAPSS, PHM2008, etc.).
- Includes various preprocessing and feature extraction methods (sliding window, normalization, RMS, kurtosis, etc.).
- Supports multiple degradation stage segmentation algorithms (3σ principle FPT, etc.).
- Offers various prediction algorithms (end-to-end prediction, single/multi-step rolling prediction, uncertainty prediction, etc.).
- Provides visualization of experimental results (confusion matrix, stage segmentation plots, prediction results, attention distribution, etc.).
- Supports importing and exporting models, datasets, experimental results, and cached files in multiple formats (CSV, PKL).
- Supports various evaluation metrics (MAE, MSE, RMSE, MAPE, PHM2012 score, NASA score, etc.).
- Allows custom component integration (easily extendable with new algorithms).
- Caches and automatically manages intermediate generated data, improving program execution speed and experimental efficiency.

## 💻    Experiment Examples
- Notebook examples: Located in the project root directory.
- Native Python examples: Available in the example folder.

## 📂    File Structure
- fastphm – Core framework code.
- doc – Detailed documentation (recommended for writing custom components).
- example – Sample experimental scripts (native Python).

## 📦    Dataset Sources
### 1. XJTU-SY Bearing Dataset 
https://biaowang.tech/xjtu-sy-bearing-datasets/
### 2. PHM2012 Bearing Dataset
https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset
### 3. C-MAPSS Turbofan Engine Dataset
https://data.nasa.gov/Aeorspace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6  
### 4. PHM2008 Dataset
https://data.nasa.gov/download/nk8v-ckry/application%2Fzip
### 5. More Datasets
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## ⚠     Important Notes
> - This framework is developed using Python 3.8.10. Compatibility issues may arise with other versions. If you encounter any problems, feel free to raise an issue.
> - When reading datasets, do not change the internal file structure of the original datasets (you may keep only partial data). Altering the file structure may lead to data reading failures.



If you find this project useful, please give it a ⭐!
If you think there’s room for improvement, feel free to submit an issue—your feedback is the greatest motivation for further updates! 😃


##### @KeyGold Studio @AndrewStudio
##### 📧 Email: andrewstudio@foxmail.com
##### 🌐 Website: http://134.175.206.112/#/home

