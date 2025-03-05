<div align="center">
    <h1>⚡ FastPHM ⚡</h1>
</div>

<div align="center"><h3>✨ 
快速上手、快速运行的 PHM 实验框架！✨</h3></div>

<div align="center">

[![GPLv3 License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Gitee star](https://gitee.com/holdenmcgorin/FastPHM/badge/star.svg?theme=dark)](https://gitee.com/holdenmcgorin/FastPHM/stargazers)
[![GitHub stars](https://img.shields.io/github/stars/holden-mcgorin/FastPHM.svg?style=social)](https://github.com/holden-mcgorin/FastPHM/stargazers)

</div>

<div align="center">

[简体中文](README.md) | [English](readme-en.md)

</div>

<div align="center">
    <a href="https://gitee.com/holdenmcgorin/FastPHM" target="_blank">Gitee</a> •
    <a href="https://github.com/holden-mcgorin/FastPHM" target="_blank">GitHub</a>
</div>

### 
> 用于快速搭建 PHM 相关实验（剩余使用寿命预测、故障诊断……），简化代码开发



## 🚀    功能简介
- 兼容多种深度学习框架进行模型搭建（PyTorch、TensorFlow、Pyro）
- 支持自动导出实验参数与结果（模型、正则化系数、迭代次数、采样次数..）
- 支持多种实验对象（轴承、涡扇发动机、电池..）
- 支持多种数据集自动导入（XJTU-SY、PHM2012、C-MAPSS、PHM2008..）
- 支持多种预处理+特征提取方法（滑动窗口、归一化、均方根、峭度..）
- 支持多种退化阶段划分算法（3σ原则FPT..）
- 支持多种预测算法（端到端预测、单/多步滚动预测、不确定性预测..）
- 支持实验结果可视化（混淆矩阵图、阶段划分图、预测结果图、注意力分布图..）
- 支持模型、数据集、实验结果、缓存的多种文件格式导入和导出（csv、pkl）
- 支持多种评价指标（MAE、MSE、RMSE、MAPE、PHM2012score、NASAscore..）
- 支持自定义组件（轻松扩展新的算法）
- 对中间生成数据进行缓存并自动管理，提升程序运行速度与实验效率

## 💻    实验示例
- Notebook示例：项目根目录
- 原生Python示例：example文件夹

## 📂    文件结构说明
- fastphm —— 框架代码
- doc —— 框架详细说明文档（编写自定义组件时建议查看）
- example —— 试验代码示例（原生python）

## 📦    数据集来源
### 1. XJTU-SY西交大轴承数据集
https://biaowang.tech/xjtu-sy-bearing-datasets/
### 2. PHM2012轴承数据集
https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset
### 3. C-MAPSS涡扇发动机数据集
https://data.nasa.gov/Aeorspace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6  
### 4. PHM2008数据集
https://data.nasa.gov/download/nk8v-ckry/application%2Fzip
### 5. 更多数据集
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

## ⚠    注意事项
> - 该框架使用Python 3.8.10编写，使用其他版本python运行可能会出现兼容性问题，若出现问题欢迎在issue提问
> - 读取数据集时，不要改变原始数据集内部文件的相对位置（可以只保留部分数据），不同的位置可能导致无法读取数据



觉得项目写的还行的大佬们点个star呗，觉得哪里写得不行的地方也欢迎issue一下，您的关注是我最大的更新动力！😀


##### @键哥工作室 @AndrewStudio
##### 📧 个人邮箱：andrewstudio@foxmail.com
##### 🌐 个人网站：http://134.175.206.112/#/home

