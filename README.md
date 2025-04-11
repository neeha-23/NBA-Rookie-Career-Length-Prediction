# NBA-Rookie-Career-Length-Prediction
====================================================================================================================================================================
					Sompalli Tirumala Neeharika - NBA Rookie Career Length Classification
====================================================================================================================================================================

**Project Overview**

This project implements multiple classification models to predict the career length of NBA rookies. The goal is to classify whether a player will have a career lasting less than 5 years or at least 5 years. The models used include:

- **K-Nearest Neighbors (KNN)**
- **Random Forest Classifier**
- **Logistic Regression**
- **Artificial Neural Network (ANN)**

The project includes feature selection, data preprocessing, hyperparameter tuning with GridSearchCV, model evaluation using 10-fold cross-validation, and performance comparison based on F1 scores.

---

### **Table of Contents**

1. Project Overview
2. Dataset
3. Implementation Details
4. Running the Program
5. Results

---

### **1. Project Overview**

The objective of this project is to analyze NBA rookie statistics and build a classification model to predict career longevity. The dataset is preprocessed by handling missing values, scaling numerical features, and engineering new features to improve prediction accuracy. 

The key steps in the project include:

- Dropping unnecessary columns (player names)
- Handling missing values
- Feature engineering (computing player efficiency)
- Data normalization (StandardScaler)
- Hyperparameter tuning using GridSearchCV
- Model evaluation using 10-fold cross-validation
- Performance comparison using F1 scores

---

### **2. Dataset**

The dataset consists of:
- **1340 rows (players)**
- **21 columns (features including performance metrics, turnovers, assists, etc.)**
- **Target variable (`TAR`)**:
  -  0 : Career length < 5 years
  -  1 : Career length >= 5 years

Preprocessing steps include:
- Handling missing values by dropping rows with NaNs.
- Feature engineering: creating an `Efficiency` metric based on player contributions.
- Scaling numerical features using StandardScaler.

---

### **3. Implementation Details**

#### **Key Functions:**

- **Data Preprocessing:**
  - drop(columns=['Name'])`: Removes irrelevant columns.
  - data.dropna()`: Removes missing values.
  - StandardScaler()`: Normalizes numerical features.

- **Model Training and Evaluation:**
  - **K-Nearest Neighbors (KNN)**: Tunes 'n_neighbors' hyperparameter.
  - **Random Forest Classifier**: Tunes 'n_estimators' and `max_depth`.
  - **Logistic Regression**: Compares 'l1', 'l2' penalties and adjusts 'C' regularization.
  - **Artificial Neural Network (ANN)**: Tunes hidden_layer_sizes and activation.
  - **GridSearchCV**: Used for hyperparameter optimization with 10-fold cross-validation.

- **Performance Metrics:**
  - **F1 Score**: Measures classification performance.
  - **Confusion Matrix**: Visualizes model predictions.
  - **ROC Curve & AUC**: Evaluates model separability.

---

### **4. Running the Program**

#### **Prerequisites:**
Ensure you have Python 3.x installed along with the required libraries:

bash
pip install pandas numpy matplotlib seaborn scikit-learn


#### **Steps to Run:**
1. Clone or download the project repository.
2. Place the dataset file (`nba.csv`) in the project directory.
3. Open a terminal and navigate to the directory.
4. Run the script using:

```bash
Sompalli-ML-project2.py
```

#### **In Google Colab:**
1. Upload `nba.csv` to Colab.
2. Upload and open Sompalli-ML-project2.
3. Run the notebook cells sequentially or you can hit cntrl+F9.

---

### **5. Results**

#### **Feature Engineering & Normalization:**
- Introduced `Efficiency` metric to improve predictive power.
- Normalized numerical features using StandardScaler, which improved distance-based models like KNN.

#### **Hyperparameter Tuning Results:**
- **KNN:** Best KNN Parameters: {'n_neighbors': 11} F1 Score: 0.8171428571428572
- **Random Forest:** Best RF Parameters: {'max_depth': 20, 'n_estimators': 200} F1 Score: 0.8169014084507042
- **Logistic Regression:** Best Logistic Regression Parameters: {'C': 0.1, 'penalty': 'l1'} F1 Score: 0.8146067415730337
- **ANN:** Best ANN Parameters: {'activation': 'tanh', 'hidden_layer_sizes': (100,)} F1 Score: 0.7900552486187845

#### **Model Performance (F1 Scores):**
- **KNN:** `0.8171`
- **Random Forest:** `0.8169`
- **Logistic Regression:** `0.8146`
- **Artificial Neural Network:** `0.7900`

**Best Model:** **Best Model: KNN with F1 Score: 0.8171428571428572**

---
