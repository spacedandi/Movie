# Low-Budget Movie Revenue Prediction 🎬

This repository contains the code and results for a thesis project focused on predicting revenue for low-budget films using machine learning and deep learning models.

---

## 📦 Dataset

The dataset used in this study is the **TMDb Movie Dataset** publicly available on Kaggle:https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

⚠️ Due to GitHub's file size limitations, the raw dataset is **not included in this repository**. Please download it manually from the link above.

---

## 🚀 How to Start

1. Download the dataset from Kaggle and place it in the project folder.
2. The overall method workflow is illustrated in `Flowchart.png`.
3. Open `movie_revenue_prediction.py`.
4. Install the required packages.
5. Run the script. Each step is clearly marked with comments.
6. The script will output model evaluation results and generate visualizations.  
   Sample outputs are also included in this repository.
7. `run_analysis.py` outputs results of the model trained on overall movie dataset.
8. `bert_overview.py` outputs results with BERT embeddings of overview features.

---

## 🧠 Project Summary

This project explores revenue prediction for **low-budget films** (budget <$15M), a high-risk and often overlooked sector in the film industry.

We compare classical machine learning models (Ridge, SVR, Random Forest, XGBoost) with **TabNet**, a deep learning model for tabular data. The dataset includes 10,623 films from TMDb (2000–2024).

Key findings:
- **XGBoost (R² = 0.84)** and **Random Forest (R² = 0.839)** outperform **TabNet (R² = 0.726)**.
- Training on only low-budget films improves prediction accuracy across all models.
- **BERT embeddings** provide limited value, likely due to noise in textual descriptions.
- **Important predictors** include budget, vote count, and genre (especially Horror, Adventure).

These results highlight the need for **tailored models** in low-budget filmmaking and provide practical insights for producers and investors.

---

## 📄 License

This project is for academic use. Please cite or credit if used in derivative work.

---
