# DS_VN: Data Valuation via Shapley Approximation

[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

**Estimate and analyze the value of training data points** using Shapley value approximation. This repository implements a systematic framework to rank data points by their contribution to model performance, remove low-value or noisy points, and compare against random removal.

## 🎯 Objectives

- Implement **Shapley value approximation** via permutation sampling to quantify the marginal contribution of each training data point.
- Train interpretable models (**Logistic Regression**) on both real-world and synthetic datasets.
- Rank data points by their **Shapley value** (data value).
- Evaluate the impact of removing:
  - Lowest-value points (targeted removal)
  - Random points (baseline removal)
- **Extension:** Inject label noise and evaluate robustness of data valuation.

## 📊 Datasets

Two main categories of classification datasets are used:

| Dataset | Type | Key Feature |
| :--- | :--- | :--- |
| **Real-world** | Small classification dataset (e.g., Breast Cancer) | Used to verify effect of **feature scaling** on data valuation |
| **Synthetic** | Controlled classification dataset | Used to study the effect of **label noise** injection and removal |

## 🧪 Experiments & Code Files

All experiments are implemented in **Jupyter Notebooks** inside the [`codes/`](./codes) folder.

### Real Data Experiments (3 notebooks)
Explores how different **feature scaling ranges** affect Shapley value distribution and data ranking:

| Notebook | Scaling Range | Purpose |
| :--- | :--- | :--- |
| `vyngoc_real_breast_ANALYSIS_SCALER_10.ipynb` | Scale data to **[-10, 10]** | Verify effect of wide-range scaling |
| `vyngoc_real_breast_ANALYSIS_SCALER_13.ipynb` | Scale data to **[-1, 3]** | Verify effect of asymmetric scaling |
| `vyngoc_real_breast_ANALYSIS_SCALER_raw.ipynb` | **Raw (unscaled)** data | Baseline without normalization |

### Synthetic Data Experiments (2 notebooks)
Focuses on **label noise injection** and the effect of removing noisy/high-value points:

| Notebook | Focus |
| :--- | :--- |
| *vyngoc_syn_ADD.ipynb* | Inject label noise → Observe change in Shapley values |
| *vyngoc_syn_REMOVE.ipynb* | Remove high-value vs low-value points → Compare final model performance |

> *Note: Exact filenames for synthetic experiments are not provided in your description – please replace placeholders with actual names.*

## 📈 Expected Outputs & Evaluation

For each experiment, you will generate:

1. **Distribution of Shapley values** (histogram/violin plot) – showing how data points are valued.
2. **Performance vs. fraction of data removed** line plot – comparing:
   - Removal of lowest-value points (Shapley-based)
   - Removal of random points
3. **Evidence of impact** – demonstrating that removing low-value points either:
   - Maintains or improves performance
   - Outperforms random removal, especially under label noise.


## 👩‍💻 Author & Contact

**Le Hong Vy Ngoc (Eiravy)**  
- GitHub: [@Eiravy](https://github.com/Eiravy)  
- Email: [hongvyngoc.le@studenti.unimi.it](mailto:hongvyngoc.le@studenti.unimi.it) 
- LinkedIn: [Le Hong Vy Ngoc](https://www.linkedin.com/in/hong-vy-ngoc-le/)

For questions, collaboration, or feedback, please feel free to reach out or open an issue on GitHub.

---

## 🚀 How to Reproduce
**Clone the repository**
   ```bash
   git clone https://github.com/Eiravy/DS_VN.git
   cd DS_VN
Run all Jupyter notebook files.
