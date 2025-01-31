# Machine Learning Tool Kit
[![Machine Learning Tool Kit](https://img.shields.io/badge/Machine%20Learning%20Tool%20Kit-PyTorch-blue?logo=pytorch)](https://github.com/Louis-Li-dev/ML_tool_kit)
- Support Pytorch Projects.
- Machine Learning Tool Kit is a Python package designed to simplify PyTorch training, data preprocessing, and various machine learning workflows. It provides a comprehensive set of tools to accelerate your machine learning projects, making development more **efficient** and **manageable**.

## Installation
```bash
pip install git+https://github.com/Louis-Li-dev/ML_tool_kit
```

## Table of Contents
- Quick Start Example: [example jupyter notebooks](https://github.com/Louis-Li-dev/ML_tool_kit/tree/main/tests)
- Support Architectures:
  - Multi-Head Attention Models
  - Convolutional Models
  - Graph Neural Networks
  - Deep Neural Networks
- Support Data Split
  - **Time Series One Cut**: Choose a date/ timestamp to split the training and testing datasets
  - **Sequences for Next-Word Prediction**: Provide x, y, and forecast_horizon to split sequences for next-word prediction or time-series tasks using transformers.
  - **Random Split**
  - **K Fold**: for advanced and better evaluation of models, k fold is prefered as users can pass in functions that are wrapped in the k fold utility function.
   
