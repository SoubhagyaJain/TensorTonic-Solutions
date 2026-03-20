<div align="center">

# 🧪 TensorTonic Solutions

**Production-Grade Implementations of Core Machine Learning Algorithms from Scratch**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/Powered_by-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![TensorTonic Framework](https://img.shields.io/badge/Platform-TensorTonic-10b981?logo=molecule&logoColor=white)](https://tensortonic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*A battle-tested, architecturally rigorous library of fundamental ML building blocks, algorithms, and deep learning architectures.*

[Explore the algorithms](#-problem-directory) • [Installation](#%EF%B8%8F-installation--setup) • [Philosophy](#-engineering-philosophy) • [Roadmap](#-future-roadmap)

</div>

---

## 📖 Overview

Implementing complex AI algorithms from scratch is the ultimate exercise in mathematical intuition and software engineering pragmatism. This repository houses my personal, **FAANG-level** solutions to the algorithmic challenges presented by [TensorTonic](https://tensortonic.com).

These are not standard textbook transcripts. They are engineered to be dropped into real-world, high-performance inference pipelines.

## 🧠 Engineering Philosophy

Every implementation in this repository adheres to strict production-grade standards:

- 🚀 **Extreme Performance:** Relentless focus on **broadcasting** and **vectorization** using NumPy. `for`-loops are eliminated wherever possible to guarantee C-level execution speeds natively in Python.
- 📐 **Numerical Stability:** Algorithms are fortified against underflow/overflow (e.g., Log-Sum-Exp tricks, epsilon clipping) ensuring deterministic and safe execution across diverse distributions.
- 🧩 **Stateless Modularity:** Clean, pure-function architectures with typed signatures. Designed specifically for composability and drop-in integration.
- 🧪 **Self-Contained Verification:** Each module is an isolated unit containing comprehensive algorithmic implementations alongside localized, rigorous test-cases.

---

## 🗂️ Problem Directory

The repository currently features **36 distinct algorithmic modules**, systematically categorized by domain:

### ⚡ Deep Learning & Neural Networks
*Core architectural components of modern representation learning.*

| Algorithm | Description | 
| :--- | :--- | 
| [**Transformer**](./transformer) | Multi-head self-attention and feed-forward layers. | 
| [**RNN Step Backward**](./rnn-step-backward) | Backpropagation Through Time (BPTT) for recurrent cells. | 
| [**ROI Pooling**](./roi-pooling) | Region of Interest pooling for object detection architectures. | 
| [**GRU Cell Forward**](./gru-cell-forward) | Forward pass for a Gated Recurrent Unit (GRU) cell. | 
| [**Leaky ReLU**](./leaky-relu) | Vectorized Leaky ReLU implementation. | 
| [**Nadam Optimizer**](./nadam-optimizer) | Perform one Nadam optimization update step. | 
| [**Sigmoid (NumPy)**](./sigmoid-numpy) | Vectorized sigmoid activation function. | 

### ⚙️ Traditional Machine Learning
*Foundational predictive modeling and sequential decision-making.*

| Algorithm | Description | 
| :--- | :--- | 
| [**Decision Tree Split**](./decision-tree-split) | Node splitting logic based on information gain/Gini impurity. | 
| [**Baseline Predictor**](./baseline-predictor) | Foundational predictive models (e.g., global average/bias). | 
| [**Gradient Descent (Quadratic)**](./gradient-descent-quadratic) | Perform gradient descent optimization on a quadratic function. | 
| [**Isotonic Calibration**](./isotonic-calibration) | Apply isotonic regression calibration. | 
| [**Value Iteration Step**](./value-iteration-step) | Perform one step of value iteration for MDPs. | 

### 🔍 NLP & Information Retrieval
*Techniques for sparse representation and text embedding.*

| Algorithm | Description | 
| :--- | :--- | 
| [**TF-IDF Vectorizer**](./tfidf-vectorizer) | Text vectorization using Term Frequency-Inverse Document Frequency. | 
| [**BM25**](./bm25) | Modern probabilistic ranking function for search engines (Okapi BM25). | 
| [**Text Chunking**](./text-chunking) | Split tokens into fixed-size chunks with overlap. | 

### 🧮 Mathematics, Probability & Linear Algebra
*The bedrock numerical primitives powering machine learning.*

| Algorithm | Description | 
| :--- | :--- | 
| [**Matrix Transpose**](./matrix-transpose) | Efficient multi-dimensional array transposition operations. | 
| [**Homogeneous Transform**](./homogeneous-transform) | 3D coordinate transformations and affine matrix multiplications. | 
| [**Bernoulli PMF**](./bernoulli-pmf) | Compute Bernoulli PMF. | 
| [**Binomial PMF/CDF**](./binomial-pmf-cdf) | Compute Binomial PMF and CDF. | 
| [**Bootstrap Mean**](./bootstrap-mean) | Compute bootstrap mean and confidence intervals. | 
| [**Chi-Squared Independence**](./chi2-independence) | Compute Chi-squared test of independence. | 
| [**Expected Value (Discrete)**](./expected-value-discrete) | Compute expected value of a discrete random variable. | 
| [**Geometric PMF & Mean**](./geometric-pmf-mean) | Compute Geometric PMF and Mean. | 
| [**Poisson PMF/CDF**](./poisson-pmf-cdf) | Compute Poisson PMF and CDF. | 
| [**Sample Variance & Std Dev**](./sample-var-std) | Compute sample variance and standard deviation. | 
| [**T-Test (One Sample)**](./t-test-one-sample) | Perform a one-sample t-test. | 

### 📏 Data Processing & Metrics
*Robust evaluation semantics and distributed feature engineering pipelines.*

| Algorithm | Description | 
| :--- | :--- | 
| [**AUC**](./auc) | Computation of the Area Under the Receiver Operating Characteristic Curve. | 
| [**Streaming Minmax**](./streaming-minmax) | Real-time tracking of minimum and maximum values in data streams. | 
| [**Batch Generator**](./batch-generator) | Randomly shuffle a dataset and yield mini-batches. | 
| [**Binning**](./binning) | Assign each value to an equal-width bin. | 
| [**ETL Dependency Orchestration**](./etl-dependency-orchestration) | Schedule ETL tasks respecting dependencies and resource limits. | 
| [**Impute Missing**](./impute-missing) | Fill NaN values in each feature column using mean or median. | 
| [**Jaccard Similarity**](./jaccard-similarity) | Compute the Jaccard similarity between two item sets. | 
| [**Min-Max Scaling**](./min-max-scaling) | Scale each column of the data matrix to the [0, 1] range. | 
| [**Robust Scaling**](./robust-scaling) | Scale values using median and interquartile range. | 
| [**Winsorization**](./winsorization) | Clip values at the given percentile bounds. | 

---

## 🛠️ Installation & Setup

Every algorithm is meticulously isolated. The entire repository is virtually dependency-free out of the box, requiring only Standard Python and `numpy`.

### 1. Clone the repository
```bash
git clone https://github.com/SoubhagyaJain/TensorTonic-Solutions.git
cd TensorTonic-Solutions
```

### 2. Prepare the Virtual Environment
```bash
# Initialize isolated environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate
# Activate (Windows)
venv\Scripts\activate

# Install extreme minimum requirements
pip install numpy
```

### 3. Usage & Testing
Navigate to any algorithmic module and explicitly review the code or run it interactively (many scripts act as their own entrypoints):

```bash
cd min-max-scaling
python main.py
```

*(Note: Advanced solutions might integrate custom test cases utilizing standard Python assertions for rapid correctness validation).*

---

## 🚀 Future Roadmap

This collection is perpetually growing. Priority architectures currently in the pipeline:

- [ ] **LLM Systems Integration:** `KV Caching`, `Rotary Positional Embeddings (RoPE)` optimizations.
- [ ] **Reinforcement Learning:** `Proximal Policy Optimization (PPO)`, `DQN / Q-Learning` logic.
- [ ] **Advanced Solvers:** Advanced Optimization techniques including `AdamW` and `L-BFGS`.
- [ ] **System Design Components:** Streaming aggregators and distributed ML scheduling algorithms.

---

<div align="center">

**Contributions & Networking**
<br/>
If you share a passion for optimizing the math beneath the abstractions, feel free to inspect the implementations.

<i>"What I cannot create, I do not understand." — Richard Feynman</i>

</div>
