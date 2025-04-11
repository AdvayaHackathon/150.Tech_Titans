import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
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

# Define column names exactly as in the dataset
column_names = [
    'Do you have a persistent cough lasting more than 2 weeks?',
    'Are you coughing up blood or blood-stained sputum?',
    'Do you experience night sweats regularly?',
    'Do you have a prolonged fever or fatigue?',
    'Have you been in close contact with a person who has TB?',
    'Have you previously been treated for TB?',
    'Have you traveled or lived in a region with high TB prevalence?',
    'Are you HIV positive?',
    'Are you undergoing immune-suppressive treatment (e.g. cancer therapy)?',
    'Are you malnourished or underweight?',
    'Do you have chronic illnesses such as diabetes?',
    'Do you have a cough with mucus or phlegm?',
    'Do you have a high fever, chills, or sweating?',
    'Are you experiencing shortness of breath?',
    'Do you feel unusually fatigued or weak?',
    'Have you recently had a respiratory infection (like flu or cold)?',
    'Do you have asthma or COPD?',
    'Do you smoke or have a smoking history?',
    'Do you have any chronic heart or lung diseases?',
    'Have you received pneumococcal or flu vaccinations?',
    'Do you experience hoarseness or wheezing?',
    'Do you have chest pain that worsens with deep breaths?',
    'Are you frequently exposed to secondhand smoke, asbestos, or radon?',
    'Do you have a family history of lung cancer?',
    'Have you been exposed to occupational hazards (e.g. mining, construction)?',
    'Have you previously had radiation therapy to the chest?',
    'Do you have any chronic lung diseases (e.g. COPD, TB scars)?',
    'Are you male?',
    'Are you above 45 years of age?',
    'likely_disease'
]


# Load dataset from CSV
try:
    df = pd.read_csv('/Users/kartikeya/Downloads/synthetic_patient_data.csv')  # Adjust path as needed
except FileNotFoundError:
    print("Error: 'multi_disease_data.csv' not found. Please provide the correct file path.")
    exit(1)

# Verify the dataset has the correct number of columns
if len(df.columns) != 30:
    print(f"Error: Expected 29 columns (29 features + 1 target), but found {len(df.columns)}.")
    print("Please ensure your CSV matches the expected format.")
    exit(1)

# Assign column names to the DataFrame
df.columns = column_names

# Verify data types and no missing values
if df[column_names[:-1]].dtypes.any() != 'int64':
    print("Warning: Some feature columns are not integers. Converting to binary (0/1).")
    df[column_names[:-1]] = df[column_names[:-1]].apply(lambda x: (x > 0).astype(int))
if df.isnull().sum().any():
    print("Error: Missing values detected. Please clean the dataset.")
    exit(1)

# Encode the target variable
le = LabelEncoder()
df['likely_disease'] = le.fit_transform(df['likely_disease'])
print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Separate features and target
X = df.drop('likely_disease', axis=1)
y = df['likely_disease']

# Check class distribution
print("Class distribution in target:")
print(pd.Series(y).value_counts(normalize=True))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance
if use_smote:
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print("After SMOTE:", pd.Series(y_train_res).value_counts())
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
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
grid_search.fit(X_train_res, y_train_res)

# Best model
best_rf = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Evaluate on test set
y_pred = best_rf.predict(X_test)
y_pred_proba = best_rf.predict_proba(X_test)

# Evaluation function for multi-class
def evaluate_model(y_true, y_pred, y_pred_proba, le):
    print("\nTest Set Evaluation:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Macro Precision: {precision_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Macro Recall: {recall_score(y_true, y_pred, average='macro'):.3f}")
    print(f"Macro F1-Score: {f1_score(y_true, y_pred, average='macro'):.3f}")
    try:
        print(f"ROC-AUC (OvR): {roc_auc_score(y_true, y_pred_proba, multi_class='ovr'):.3f}")
    except ValueError:
        print("ROC-AUC not computed due to class imbalance or insufficient samples.")

evaluate_model(y_test, y_pred, y_pred_proba, le)

# Feature importance
feature_importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save the model and label encoder
joblib.dump(best_rf, 'disease_predictor.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("\nModel saved as 'disease_predictor.pkl' and label encoder as 'label_encoder.pkl'")

# Interactive prediction function
def predict_disease():
    print("\n=== Disease Diagnosis Predictor ===")
    print("Note: This is not a substitute for professional medical diagnosis.")
    print("Enter patient details (0 for No, 1 for Yes):")
    features = {}
    for col in X.columns:
        while True:
            try:
                value = int(input(f"{col}: "))
                if value in [0, 1]:
                    features[col] = value
                    break
                else:
                    print("Please enter 0 or 1.")
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")
    
    # Create input DataFrame
    input_df = pd.DataFrame([features], columns=X.columns)
    
    # Load model and label encoder
    try:
        model = joblib.load('disease_predictor.pkl')
        le = joblib.load('label_encoder.pkl')
    except FileNotFoundError:
        print("Error: Model or label encoder file not found.")
        return
    
    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    # Decode prediction
    predicted_disease = le.inverse_transform([prediction])[0]
    
    # Output result
    print("\nPrediction Result:")
    print(f"Most likely disease: {predicted_disease}")
    print("Class probabilities:")
    for cls, prob in zip(le.classes_, probabilities):
        print(f"{cls}: {prob:.2%}")
    
    if predicted_disease != 'None':
        print(f"Recommendation: Seek immediate medical evaluation for {predicted_disease}.")
    else:
        print("Recommendation: Monitor symptoms; consult a doctor if symptoms persist.")
    
    # Highlight key factors
    important_features = feature_importance.head(3).index
    present_features = [f for f in important_features if features[f] == 1]
    if present_features and predicted_disease != 'None':
        print(f"Key risk factors detected: {', '.join(present_features)}")
    elif predicted_disease != 'None':
        print("No top risk factors detected, but medical evaluation is still recommended.")

# Run prediction interactively
while True:
    predict_disease()
    again = input("\nWould you like to make another prediction? (yes/no): ").lower()
    if again != 'yes':
        print("Exiting predictor.")
        break