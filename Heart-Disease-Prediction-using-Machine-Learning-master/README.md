# Heart Disease Prediction using Machine Learning
### By Daivik Reddy

## Project Overview
Heart disease is a leading cause of mortality worldwide. Early detection and prevention are crucial for reducing its impact. This project uses machine learning to predict the presence of heart disease in patients based on their medical attributes.

Good data-driven systems for predicting heart diseases can improve the entire research and prevention process, helping more people live healthier lives. This project demonstrates how machine learning can make accurate predictions for heart disease diagnosis.

## Dataset Information
- **Source**: [UCI Heart Disease Dataset on Kaggle](https://www.kaggle.com/ronitf/heart-disease-uci)
- **Features**: The dataset contains 13 medical attributes including:
  - Age
  - Sex
  - Chest pain type
  - Resting blood pressure
  - Serum cholesterol
  - Fasting blood sugar
  - Resting ECG results
  - Maximum heart rate achieved
  - Exercise-induced angina
  - ST depression induced by exercise
  - Slope of the peak exercise ST segment
  - Number of major vessels colored by fluoroscopy
  - Thalassemia
- **Target Variable**: Binary (1 = heart disease present, 0 = heart disease absent)

## Machine Learning Models Implemented
This project compares 8 different classification algorithms:

1. Logistic Regression (Scikit-learn)
2. Naive Bayes (Scikit-learn)
3. Support Vector Machine (Linear) (Scikit-learn)
4. K-Nearest Neighbors (Scikit-learn)
5. Decision Tree (Scikit-learn)
6. Random Forest (Scikit-learn) - **Best performer with 95% accuracy**
7. XGBoost (Scikit-learn)
8. Artificial Neural Network with 1 Hidden layer (Keras)

## How to Use This Project

### Prerequisites
- Python 3.6+
- Jupyter Notebook
- Required libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, keras, tensorflow

### Installation
1. Clone this repository:
   ```
   git clone https://github.com/Daivik1520/Heart-Disease-Prediction-using-Machine-Learning.git
   cd Heart-Disease-Prediction-using-Machine-Learning
   ```

2. Install required packages:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow
   ```

3. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

4. Open `Heart_disease_prediction.ipynb` in Jupyter Notebook

### Using the Model
1. The notebook is divided into sections:
   - Data preprocessing and exploration
   - Feature selection and engineering
   - Model training and evaluation
   - Results comparison

2. To use the trained model for prediction:
   - Run all cells in the notebook
   - The best performing model (Random Forest) is saved and can be used for predictions
   - Input patient data in the same format as the dataset
   - Use the model to predict the likelihood of heart disease

3. For new predictions, use the code snippet in the final section of the notebook.

## Results
- The Random Forest model achieved the highest accuracy of 95%
- Feature importance analysis reveals the most significant medical indicators for heart disease prediction
- Detailed performance metrics (precision, recall, F1-score) are provided in the notebook

## Future Improvements
- Implement cross-validation for more robust model evaluation
- Deploy the model as a web application for easier access
- Incorporate additional medical features for potentially higher accuracy
- Explore deep learning approaches for improved performance

