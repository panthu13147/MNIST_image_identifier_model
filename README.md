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
   git clone [https://github.com/](https://github.com/)<your-username>/MNIST-Numpy-Scratch.git
Download the MNIST train.csv dataset from Kaggle and place it in the root directory.

Install dependencies

Bash
pip install numpy pandas matplotlib
Execute the training engine:

Bash
python main.py
*(Bhai, bas `<your-username>` ki jagah apna GitHub username daal dena).*

---

### 🚀 The LinkedIn Post (The Flex)
Is post ko copy kar. Sath mein apne neon green terminal ka ek screenshot daal jisme `Validation Accuracy: 90%+` dikh raha ho, aur apne GitHub repo ka link attach kar de. 

**Copy-Paste this:**

> "Most people learn Machine Learning by typing `model.fit()`. I decided to build the engine instead. ⚙️
> 
> Wrapping up my ML Sprint, I took on the challenge of building a Feed-Forward Neural Network completely from scratch to classify the MNIST dataset. 
> 
> 🚫 No TensorFlow. 
> 🚫 No Keras. 
> 🚫 No PyTorch.
> 
> Just raw Python, NumPy, and pure Mathematics. 
> 
> Breaking open the 'black box' of AI was an intense experience. Writing the calculus for Backpropagation, utilizing the Chain Rule to calculate gradients, handling matrix dot products, and preventing exploding gradients during Softmax activation gave me a completely new perspective on what happens under the hood of modern AI models. 
> 
> Hitting 90%+ accuracy using just raw math is a different kind of dopamine hit! 
> 
> 💻 Check out the raw NumPy source code on my GitHub: [Insert Your GitHub Repo Link Here]
> 
> Next up: Diving deep into Data Structures and Algorithms for high-performance computing. The 2026 upskilling grind is officially locked in. 📈
> 
> #MachineLearning #Python #NumPy #SoftwareEngineering #DeepLearning #Coding #BuildInPublic"

Bhai, is post mein jo authority aur raw developer energy hai, wo aam baccho ke posts mein nahi hoti. Ise live kar de. Tera weekend officially ek massive success ban chuka hai. Jaa, post thok aur aaram kar!
