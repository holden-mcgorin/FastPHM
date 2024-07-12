# 故障预测框架

## 👻    功能简介
- 简化剩余使用寿命预测、故障诊断的代码编写，附带示例代码
- 对中间生成数据进行缓存并自动管理，提升程序运行速度与实验效率
- 兼容多种深度学习框架进行模型搭建（PyTorch、TensorFlow、Pyro）
- 支持自动导出实验参数与结果（模型、正则化系数、迭代次数、采样次数等）
- 支持多种数据集自动导入（XJTU-SY、PHM2012、IMS）
- 支持多种轴承退化特征提取（均方根、峭度）
- 支持多种轴承阶段划分算法（3σ原则FPT、10倍振幅EoL）
- 支持多种预测算法（单/多步滚动预测、不确定性预测）
- 支持实验结果可视化（混淆矩阵图、寿命周期划分图、loss图、预测结果图）
- 支持多种评价指标（MSE、MAPE、PHM2012score）
- 支持自定义组件（轻松扩展新的算法）


## 💡    安装方法
### 使用git库远程安装
1. pip install git+https://gitee.com/holdenmcgorin/RULFramework
### 使用源代码安装
1. 下载源代码 或 git clone https://gitee.com/holdenmcgorin/RULFramework.git
2. 进入此项目根目录
3. pip install .

## 💻    代码示例
- notebook示例：项目根目录
- 原生python示例：example文件夹

## 📂    文件结构说明
- rulframework —— 框架代码
- doc —— 框架详细说明文档（编写自定义组件时建议查看）
- example —— 试验代码示例（原生python）

## ⚠    注意事项
- 读取数据集时，不要改变原始数据集内部文件的相对位置（可以只保留部分数据，但不要改变相对位置），可能导致无法读取数据


觉得项目写的还行的大佬们点个star呗，觉得哪里写得不行的地方也欢迎issue一下，您的关注是我最大的更新动力！😀


##### @键哥工作室 @AndrewStudio
##### 个人邮箱：andrewstudio@foxmail.com
##### 个人网站：http://139.9.192.174/#/home

