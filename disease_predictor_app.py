import streamlit as st
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import re



# Function to write answers to CSV
def write_to_csv(file_name, questions, answers):
    with open(file_name, 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerow(questions)
        writer.writerow(answers)

# Prediction function (adapted from your script)
def predict_disease(answers, feature_names):
    # Load model and label encoder
    try:
        model = joblib.load('disease_predictor.pkl')
        le = joblib.load('label_encoder.pkl')
    except FileNotFoundError:
        st.error("Model or label encoder file not found.")
        return None

    # Create input DataFrame
    input_df = pd.DataFrame([answers], columns=feature_names)

    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    predicted_disease = le.inverse_transform([prediction])[0]

    # Format probabilities
    prob_dict = {cls: float(prob) for cls, prob in zip(le.classes_, probabilities)}

    # Recommendation
    recommendation = (
        f"Seek immediate medical evaluation for {predicted_disease}."
        if predicted_disease != 'None'
        else "Monitor symptoms; consult a doctor if symptoms persist."
    )

    # Feature importance
    feature_importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    important_features = feature_importance.head(3).index.tolist()
    present_features = [f for f in important_features if input_df[f].iloc[0] == 1]

    return {
        "predicted_disease": predicted_disease,
        "probabilities": prob_dict,
        "recommendation": recommendation,
        "key_risk_factors": present_features if present_features and predicted_disease != 'None' else []
    }

# Main Streamlit app
def main():
    # Define questions
    disease_questions = [
        "Do you have a persistent cough lasting more than 2 weeks?",
        "Are you coughing up blood or blood-stained sputum?",
        "Do you experience night sweats regularly?",
        "Do you have a prolonged fever or fatigue?",
        "Have you been in close contact with a person who has TB?",
        "Have you previously been treated for TB?",
        "Have you traveled or lived in a region with high TB prevalence?",
        "Are you HIV positive?",
        "Are you undergoing immune-suppressive treatment (e.g. cancer therapy)?",
        "Are you malnourished or underweight?",
        "Do you have chronic illnesses such as diabetes?",
        "Do you have a cough with mucus or phlegm?",
        "Do you have a high fever, chills, or sweating?",
        "Are you experiencing shortness of breath?",
        "Do you feel unusually fatigued or weak?",
        "Have you recently had a respiratory infection (like flu or cold)?",
        "Do you have asthma or COPD?",
        "Do you smoke or have a smoking history?",
        "Do you have any chronic heart or lung diseases?",
        "Have you received pneumococcal or flu vaccinations?",
        "Do you experience hoarseness or wheezing?",
        "Do you have chest pain that worsens with deep breaths?",
        "Are you frequently exposed to secondhand smoke, asbestos, or radon?",
        "Do you have a family history of lung cancer?",
        "Have you been exposed to occupational hazards (e.g. mining, construction)?",
        "Have you previously had radiation therapy to the chest?",
        "Do you have any chronic lung diseases (e.g. COPD, TB scars)?",
        "Are you male?",
        "Are you above 45 years of age?"
    ]

    # Title of the app
    st.title("Answer these questions to determine your disease")

    # Initialize session state
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    # Create form for answers
    if not st.session_state.submitted:
        with st.form(key="question_form"):
            answers = []
            for i, question in enumerate(disease_questions, 1):
                answer = st.radio(f"Question {i}: {question}", options=["Yes", "No"], key=f"q{i}")
                answers.append(1 if answer == "Yes" else 0)
            submit_button = st.form_submit_button(label="Submit Answers")

            if submit_button:
                # Write to CSV for debugging (optional)
                write_to_csv("main_V2_data.csv", disease_questions, answers)
                st.session_state.submitted = True
                # Run prediction
                prediction = predict_disease(answers, disease_questions)
                if prediction:
                    st.session_state.prediction = prediction
                    st.success("Prediction complete!")
                else:
                    st.error("Prediction failed. Please try again.")

    # Display prediction
    if st.session_state.submitted and st.session_state.prediction:
        pred = st.session_state.prediction
        st.write("### Prediction Result:")
        st.write(f"**Most likely disease**: {pred['predicted_disease']}")
        st.write("**Class probabilities**:")
        for cls, prob in pred['probabilities'].items():
            st.write(f"{cls}: {prob:.2%}")
        if pred['predicted_disease'] == "tuberculosis":
            st.write("Treatment: Long-term antibiotics (e.g., isoniazid, rifampin) for 6-9 months; directly observed therapy (DOT) to ensure compliance.\n\nCommon Causes: Infection by Mycobacterium tuberculosis, spread through airborne droplets; risk factors include close contact with infected individuals, weakened immune systems (e.g., HIV), and living in high-prevalence areas.")
        
        elif pred['predicted_disease'] == "pneumonia":
            st.write("Treatment: Antibiotics for bacterial pneumonia (e.g., amoxicillin); antivirals or antifungals for viral/fungal cases; oxygen therapy and fluids for severe cases.\n\nCommon Causes: Bacterial (Streptococcus pneumoniae), viral (e.g., influenza), or fungal infections; risk factors include smoking, chronic lung diseases, and recent respiratory infections.")

        elif pred['predicted_disease'] == "lung_cancer":
            st.write("Treatment: Surgery, chemotherapy, radiation, targeted therapy, or immunotherapy, depending on stage and type (small cell or non-small cell).\n\nCommon Causes: Smoking (primary cause), exposure to radon, asbestos, or secondhand smoke; family history and occupational hazards (e.g., mining) increase risk.")

    
    st.title("Do a deeper analysis with images?")


            # Streamlit app
    st.write("Image Upload App")
    with st.form(key="form 3"):
    # Image upload widget
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader", accept_multiple_files=True)

        submit_button3 = st.form_submit_button(label="Submit Image")
        if submit_button3:
            st.write("Loading... (max 10s)")
        # Check if an image has been uploaded
            if uploaded_file is not None:
                for i in uploaded_file:
                # Display the uploaded image
                    image = Image.open(i)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Explicit Submit button
                    if submit_button3:

                                    # Set up folder to save images
                        SAVE_DIR = "./test_imgs"
                        if not os.path.exists(SAVE_DIR):
                            os.makedirs(SAVE_DIR)
                        # Define the file path
                        file_path = os.path.join(SAVE_DIR, i.name)
                        # Save the image
                        with open(file_path, "wb") as f:
                            f.write(i.getbuffer())
                        st.success(f"Image saved successfully to {file_path}")
            else:
                st.write("Please upload an image to proceed.")


    
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        import tensorflow as tf

                    
        # Load model
        model = load_model('xray_densenet.keras')
        print("✅ Model loaded")

        # Define class labels (from your training generator)
        class_labels = ['Normal', 'Pneumonia', 'Tuberculosis']
        folder_path = "./test_imgs"
        files = os.listdir(folder_path)

        for file in files:
            
        # Load and preprocess the image
            img_path = f'.//test_imgs//{file}' 
            img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
            img_array = img_array / 255.0  # normalize

            # Predict
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Output result
            print(f"🔍 Prediction: {class_labels[predicted_class]} ({confidence:.2%} confidence)")

            
            
            if class_labels[predicted_class] == 'Tuberculosis':
                st.write(f"The image is predicted to be Tuberculosis with confidence: {confidence:.2%}")

            elif class_labels[predicted_class] == 'Pneumonia':
                st.write(f"The image is predicted to be Pnemonia with confidence: {confidence:.2%}")
                
            elif class_labels[predicted_class] == 'Lung_cancer':
                st.write(f"The image is predicted to be  Lung cancer with confidence: {confidence:.2%}")
            else:
                st.write(f"The image is shows a health lung with confidence: {confidence:.2%}")

    with st.form(key="form 4"):

        submit_button4 = st.form_submit_button(label="Delete added Image?")

    # Check if an image has been uploaded
        if submit_button4:
            list1 = os.listdir("./test_imgs")
            for i in list1:
                os.remove(os.path.join("./test_imgs", i))



             

def save_uploaded_images(uploaded_files, output_dir="uploaded_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_paths = []
    for file in uploaded_files:
        image = Image.open(file)
        # Save with a unique name (e.g., timestamp + original name)
        file_path = os.path.join(output_dir, f"{time.time()}_{file.name}")
        image.save(file_path)
        image_paths.append(file_path)
    return image_paths


main()