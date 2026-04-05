# 🧠 MNIST Digit Classifier - Pure NumPy (Built From Scratch)

![Python](https://img.shields.io/badge/Python-3.9-blue.svg?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Raw_Math-brightgreen.svg?style=for-the-badge&logo=numpy)
![Machine Learning](https://img.shields.io/badge/ML-From_Scratch-red.svg?style=for-the-badge)

## 📌 The "Why?"
The tech world is full of engineers who can type `model.fit()` using TensorFlow or PyTorch without understanding the underlying math. I wanted to escape the "black-box" approach. 

This project is a fully functional, Feed-Forward Neural Network built **entirely from scratch** using only raw Python and NumPy matrices. No high-level ML libraries were used. 

## 🏗️ Neural Network Architecture
- **Input Layer:** 784 nodes (Flattened 28x28 MNIST pixel images)
- **Hidden Layer:** 128 nodes (with Sigmoid Activation)
- **Output Layer:** 10 nodes (with Softmax Activation for probability distribution)
- **Loss Function:** Categorical Cross-Entropy
- **Optimization Algorithm:** Mini-Batch Gradient Descent

## ⚙️ Core Engineering Challenges Solved
- Implemented **Forward Propagation** using vectorized dot products for maximum efficiency.
- Derived and coded the **calculus for Backpropagation** (chain rule) manually to compute weight and bias gradients.
- Implemented dynamic **One-Hot Encoding** for accurate array-to-array loss calculation.
- Prevented mathematical overflow (NaN) errors during Softmax execution by stabilizing the exponential outputs.

## 📈 Results
Achieved **90%+ Validation Accuracy** within 100 epochs, proving the mathematical logic holds up against real-world data.

## 💻 How to Run This Locally
1. Clone this repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<panthu13147>/MNIST-Numpy-Scratch.git
Download the MNIST train.csv dataset from Kaggle and place it in the root directory.

Install dependencies

Bash
pip install numpy pandas matplotlib
Execute the training engine:

Bash
python main.py
*(Bhai, bas `<your-username>` ki jagah apna GitHub username daal dena).*

---

