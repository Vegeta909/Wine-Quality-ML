🍷 Wine Quality Classification Using Machine Learning

📌 Overview

This project leverages machine learning techniques to classify wine into "good" and "bad" quality categories based on its physicochemical properties. By analyzing features such as alcohol content, pH, acidity, and density, the project aims to automate wine quality assessment with high accuracy and consistency.

🚀 Features

Data Preprocessing & Feature Engineering
Comparative Analysis of Machine Learning Models (KNN, SVM, Naïve Bayes, Random Forest, XGBoost, Logistic Regression)
Hyperparameter Tuning using Grid Search
Evaluation Metrics: Accuracy, Precision, Recall, and F1 Score
Data Visualizations (Feature Correlation, Model Performance, and Feature Importance)

📂 Dataset

The dataset is publicly available on the UCI Machine Learning Repository:
Red Wine Dataset: 1599 instances
Features: 11 input features and 1 output feature (quality)

🔧 Installation & Setup

1️⃣ Clone the Repository

git clone <repository_link>
cd Wine-Quality-Classification

2️⃣ Set Up Virtual Environment (Recommended)

python -m venv venv

source venv/bin/activate  # On Mac/Linux  
venv\Scripts\activate     # On Windows

3️⃣ Install Dependencies

pip install jupyter pandas numpy matplotlib seaborn scikit-learn xgboost

4️⃣ Run the Jupyter Notebook

jupyter notebook

Open Wine_Quality.ipynb and run the cells.

🏗 Project Workflow

1️⃣ Data Preprocessing

Handling missing values
Normalizing features
Analyzing correlations

2️⃣ Model Training

KNN, SVM, Naïve Bayes, Random Forest, XGBoost, Logistic Regression
Splitting dataset into training & testing sets

3️⃣ Hyperparameter Tuning

Optimizing model parameters using Grid Search

4️⃣ Model Evaluation

Accuracy, Precision, Recall, F1-score
Confusion Matrices & Performance Graphs

📊 Results

Model
Accuracy

Random Forest
89.3%

K-Nearest Neighbors (KNN)
87.2%

Logistic Regression
87.0%

Support Vector Classifier (SVC)
86.8%

Decision Tree
86.4%

Gaussian Naïve Bayes
83.3%

📌 Random Forest achieved the highest accuracy (89.3%), making it the best model for wine quality classification.

📈 Visualizations

Data Distribution (Histograms, Boxplots)
Feature Importance (Random Forest & XGBoost)
Model Performance (Confusion Matrices, Accuracy Graphs)

🤝 Contributing

Contributions are welcome! Feel free to open an issue.

📩 Contact

For questions or feedback, feel free to reach out:

Name: Arnav Singh Bisht
Email: arnavsb909@gmail.com
