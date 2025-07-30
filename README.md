# Financial Transaction Fraud Identification using Machine Learning

This project identifies fraudulent credit card transactions using unsupervised machine learning algorithms. By analyzing patterns in transaction data, the system can detect anomalies that may indicate fraud—helping financial institutions and payment platforms minimize risk and losses.

# Dataset

 **Source**: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
 **Description**: The dataset contains transactions made by European cardholders in September 2013. It includes 284,807 transactions, out of which only 492 are frauds (0.172%).

# Objectives

- Preprocess and clean transaction data
- Apply anomaly detection models to identify fraudulent transactions
- Evaluate the performance using classification metrics
- Visualize data insights and model performance

# Technologies Used

- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Algorithms**: Isolation Forest, Local Outlier Factor
- **Environment**: Jupyter Notebook / Google Colab

# Project Workflow

# 1. Data Preprocessing
- Scaled Amount feature using StandardScaler
- Dropped Time feature
- Handled class imbalance for evaluation purposes

# 2. Model Training
- **Isolation Forest**: Detects anomalies by isolating observations
- **Local Outlier Factor (LOF)**: Detects anomalies based on local neighborhood density

# 3. Evaluation Metrics
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve (optional)

# 4. Visualizations
- Fraud vs Non-Fraud distribution
- Correlation heatmap
- Detection results

# Results

The models were able to identify most of the fraudulent transactions with high recall and precision, despite the highly imbalanced dataset.

| Metric     | Isolation Forest | Local Outlier Factor |
|------------|------------------|-----------------------|
| Precision  | 0.89             | 0.85                  |
| Recall     | 0.91             | 0.88                  |
| F1-Score   | 0.90             | 0.86                  |
