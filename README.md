# Heart Disease Risk Prediction

## Project Overview

This project performs end-to-end analysis on a heart disease risk dataset, including data cleaning, exploratory analysis, feature engineering, model development, and deployment preparation. The goal is to build a machine learning model that can predict the risk of heart disease based on various health indicators and demographic information.

## Dataset

The dataset (`heart_disease_risk_dataset_earlymed.csv`) contains various health indicators and demographic information, with a binary target variable indicating heart disease risk. The dataset initially contained 70,000 records with 19 features, which was reduced to 63,755 records after removing duplicates.

### Features

- **Symptoms**: Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations, Dizziness, Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea
- **Medical Conditions**: High_BP, High_Cholesterol, Diabetes
- **Lifestyle Factors**: Smoking, Obesity, Sedentary_Lifestyle, Chronic_Stress
- **Demographics**: Gender, Age, Family_History
- **Target Variable**: Heart_Risk (0: No risk, 1: Risk)

## Project Structure

The project is organized as follows:

```
heart_disease_risk_prediction/
├── heart_disease_analysis.ipynb     # Jupyter notebook with complete analysis
├── cleaned_heart_disease_data.csv        # Cleaned dataset
├── heart_disease_risk_dataset_earlymed.csv      # dataset
├── best_heart_risk_model.pkl        # Saved best model
├── selected_features.pkl            # List of selected features for the model
├── app.py                           # Flask application for model deployment
├── README.md                        # Project documentation
└── templates/                       # HTML templates for the Flask app
    └── index.html                   # Home page template
```

## Methodology

### 1. Data Cleaning and Preprocessing

- Removed duplicate records (6,245 duplicates found)
- Checked for missing values (none found)
- Verified data consistency and format
- Examined potential outliers

### 2. Exploratory Data Analysis

- Performed univariate analysis on key variables
- Conducted bivariate analysis to understand relationships between variables
- Created visualizations including histograms, boxplots, and correlation matrices
- Identified patterns and relationships in the data

### 3. Feature Engineering and Selection

- Created age category feature (Young, Middle_Aged, Senior)
- Used correlation analysis to identify relevant features
- Applied Recursive Feature Elimination (RFE) for feature selection
- Combined highly correlated features and RFE selected features

### 4. Model Development and Evaluation

- Implemented and compared three machine learning algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Performed hyperparameter tuning using GridSearchCV
- Evaluated models using accuracy, precision, recall, and F1-score
- Selected the best performing model (Gradient Boosting)

### 5. Deployment Preparation

- Exported the cleaned dataset
- Saved the best model and selected features
- Created a Flask web application for model deployment

## Results

The Gradient Boosting model achieved the best performance with:
- Accuracy: 0.9924
- Precision: 0.9908
- Recall: 0.9938
- F1-Score: 0.9923

## Deployment

The model is deployed as a Flask web application that allows users to input patient information and receive a heart disease risk prediction.

### Running the Application

1. Install the required dependencies:
```
pip install flask pandas numpy scikit-learn joblib
```

2. Run the Flask application:
```
python app.py
```

3. Access the application in a web browser at `http://localhost:5000`

## Future Work

- Collect more diverse data to improve model generalization
- Implement more advanced feature engineering techniques
- Explore deep learning approaches
- Enhance the web interface with more detailed risk assessments
- Add explanations for predictions using techniques like SHAP values

## Video presentation 
https://www.canva.com/design/DAGpm4RGlEo/u7eNfXZK6vzN173l66mW6g/edit?utm_content=DAGpm4RGlEo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton


## paper
https://drive.google.com/file/d/1I8peHdf4bUo_30cbdz1A4VZuwELpm1nn/view?usp=drivesdk

## Author

This project was developed by Mohamed Abdelkader Ragab 

