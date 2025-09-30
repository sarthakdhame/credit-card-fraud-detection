# 🔒 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using Decision Tree Classifier with SMOTE balancing technique.

## 📊 Project Overview

This project implements an end-to-end machine learning pipeline for credit card fraud detection, addressing the challenge of highly imbalanced datasets in financial fraud detection.

## 🎯 Key Features

- **Data Preprocessing**: Standardization, duplicate removal, feature engineering
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)
- **Model Training**: Decision Tree Classifier with hyperparameter optimization
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1-Score)
- **Model Persistence**: Trained model saved for deployment

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Imbalanced-learn** - SMOTE implementation
- **Seaborn & Matplotlib** - Data visualization
- **Joblib** - Model serialization

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── main.ipynb              # Complete ML pipeline notebook
├── credit_card_model.pkl   # Trained model file
├── creditcard.csv          # Dataset (not included - download separately)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib joblib
```

### Dataset

Download the Credit Card Fraud Detection dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and place it as `creditcard.csv` in the project directory.

### Running the Project

1. Clone this repository
2. Install required packages
3. Download the dataset
4. Open and run `main.ipynb` in Jupyter Notebook

## 📈 Model Performance

### Before SMOTE (Imbalanced Data)
- **Dataset**: ~275K transactions
- **Fraud Cases**: 473 (0.17%)
- **Normal Cases**: 275K+ (99.83%)

### After SMOTE (Balanced Data)
- **Balanced Dataset**: Equal representation of both classes
- **Improved Recall**: Better detection of fraudulent transactions
- **Model**: Decision Tree Classifier

### Results
```
Logistic Regression:
- Accuracy: 99.93%
- Precision: 89.06%
- Recall: 62.64%
- F1-Score: 73.55%

Decision Tree Classifier:
- Accuracy: 99.90%
- Precision: 67.00%
- Recall: 73.63%
- F1-Score: 70.16%
```

## 🔍 Key Insights

1. **Class Imbalance Challenge**: Original dataset heavily skewed (99.83% normal transactions)
2. **SMOTE Effectiveness**: Significantly improved model's ability to detect fraud
3. **Feature Importance**: PCA-transformed features (V1-V28) + Amount
4. **Trade-off Analysis**: Balance between precision and recall for fraud detection

## 📋 Methodology

### 1. Data Exploration
- Dataset shape analysis
- Missing value detection
- Class distribution visualization

### 2. Data Preprocessing
- StandardScaler for Amount feature
- Time column removal
- Duplicate removal

### 3. Class Balancing
- SMOTE application for minority class oversampling
- Balanced dataset creation

### 4. Model Training
- Train-test split (80-20)
- Multiple algorithm comparison
- Model selection based on performance metrics

### 5. Model Evaluation
- Confusion matrix analysis
- Precision-Recall trade-off
- Cross-validation scores

## 🎯 Business Impact

- **Financial Security**: Helps prevent fraudulent transactions
- **Cost Reduction**: Reduces manual review overhead
- **Customer Trust**: Enhances security confidence
- **Real-time Detection**: Model ready for production deployment

## 🔮 Future Enhancements

- [ ] Ensemble methods (Random Forest, XGBoost)
- [ ] Deep learning approaches (Neural Networks)
- [ ] Real-time prediction API
- [ ] Model monitoring and drift detection
- [ ] Feature engineering optimization

## 📚 Learning Outcomes

This project demonstrates:
- End-to-end ML pipeline development
- Handling imbalanced datasets
- Model evaluation and selection
- Production-ready model preparation

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset provided by [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- SMOTE technique from imbalanced-learn library
- Scikit-learn for machine learning algorithms

---

**Note**: This project was developed as a learning exercise following machine learning best practices and tutorials. The implementation focuses on understanding fraud detection techniques and building a complete ML pipeline."# credit-card-fraud-detection" 
