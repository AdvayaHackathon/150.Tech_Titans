import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import SMOTE; use class weights if it fails
try:
    from imblearn.over_sampling import SMOTE
    use_smote = True
    print("SMOTE imported successfully.")
except ImportError:
    print("SMOTE not available. Using class weights instead.")
    use_smote = False

# Load dataset from CSV
try:
    df = pd.read_csv('/Users/kartikeya/Downloads/tuberculosis_augmented_2.csv')
except FileNotFoundError:
    print("Error: 'tuberculosis_augmented_2.csv' not found. Please provide the correct file path.")
    exit(1)

# Define column names based on your dataset (15 columns: 14 features + 1 target)
column_names = [
    'Do you have a persistent cough lasting more than 2 weeks?',
    'Are you coughing up blood or blood-stained sputum?',
    'Do you experience night sweats regularly?',
    'Have you had an unexplained weight loss recently?',
    'Do you have a prolonged fever or fatigue?',
    'Have you been in close contact with a person who has TB?',
    'Have you previously been treated for TB?',
    'Have you traveled or lived in a region with high TB prevalence?',
    'Are you HIV positive?',
    'Are you undergoing immune-suppressive treatment (e.g. cancer therapy)?',
    'Are you malnourished or underweight?',
    'Do you have chronic illnesses such as diabetes?',
    'Are you male?',
    'Are you above 45 years of age?',  # Comma added here
    'Has Tuberculosis'
]

# Verify the dataset has the correct number of columns
if len(df.columns) != 15:
    print(f"Error: Expected 15 columns (14 features + 1 target), but found {len(df.columns)}.")
    print("Please ensure your CSV matches the expected format.")
    exit(1)

# Assign column names to the DataFrame
df.columns = column_names

# Verify data types and no missing values
if df[column_names].dtypes.any() != 'int64':
    print("Warning: Some columns are not integers. Converting to binary (0/1).")
    df[column_names] = df[column_names].apply(lambda x: (x > 0).astype(int))
if df.isnull().sum().any():
    print("Error: Missing values detected. Please clean the dataset.")
    exit(1)

# Separate features and target
X = df.drop('Has Tuberculosis', axis=1)
y = df['Has Tuberculosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance
if use_smote:
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    class_weight = None
else:
    X_train_res, y_train_res = X_train, y_train
    class_weight = 'balanced'

# Train and tune Random Forest
print("Training Random Forest with hyperparameter tuning...")
rf = RandomForestClassifier(random_state=42, class_weight=class_weight)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

# Best model
best_rf = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate on test set
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

# Evaluation function
def evaluate_model(y_true, y_pred, y_pred_proba):
    print("\nTest Set Evaluation:")
    print(classification_report(y_true, y_pred))
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall: {recall_score(y_true, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.3f}")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_pred_proba):.3f}")

evaluate_model(y_test, y_pred, y_pred_proba)

# Feature importance
feature_importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the model
joblib.dump(best_rf, 'tb_predictor.pkl')
print("\nModel saved as 'tb_predictor.pkl'")

# Interactive prediction function
def predict_tb():
    print("\n=== TB Diagnosis Predictor ===")
    print("Enter patient details (0 for No, 1 for Yes):")
    features = {}
    for col in X.columns:
        while True:
            try:
                value = int(input(f"{col.replace('_', ' ')}: "))
                if value in [0, 1]:
                    features[col] = value
                    break
                else:
                    print("Please enter 0 or 1.")
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")
    
    # Create input DataFrame
    input_df = pd.DataFrame([features], columns=X.columns)
    
    # Load model and predict
    model = joblib.load('tb_predictor.pkl')
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]
    
    # Adjust threshold for higher recall (set to 0.4 to catch more TB cases)
    threshold = 0.4
    adjusted_prediction = 1 if probability > threshold else 0
    
    # Output result
    print("\nPrediction Result:")
    if adjusted_prediction == 1:
        print(f"The patient is likely to have TB (Probability: {probability:.2%}).")
        print("Recommendation: Seek immediate medical evaluation, including sputum tests or chest X-ray.")
    else:
        print(f"The patient is unlikely to have TB (Probability of TB: {probability:.2%}).")
        print("Recommendation: Monitor symptoms; consult a doctor if symptoms persist.")
    
    # Highlight key factors
    important_features = feature_importance.head(3).index
    present_features = [f for f in important_features if features[f] == 1]
    if present_features and adjusted_prediction == 1:
        print(f"Key risk factors detected: {', '.join([f.replace('_', ' ') for f in present_features])}")

# Run prediction interactively
while True:
    predict_tb()
    again = input("\nWould you like to make another prediction? (yes/no): ").lower()
    if again != 'yes':
        print("Exiting predictor.")
        break