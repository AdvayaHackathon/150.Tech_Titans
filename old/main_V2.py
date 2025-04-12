import csv
import streamlit as st


def write_to_csv(file_name, csv_data):
    with open(file_name, 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerows(csv_data)

def main(the_file_name):

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

    # Define a list of questions
    questions = disease_questions

    # Initialize a list to store answers
    answers = []

    # Create a form to collect all answers at once
    with st.form(key="question_form"):
        # Loop through each question and create a text input
        i = 0
        for question in questions:
            i += 1
            answer = st.radio(f"Question {i}: {question}", options=["yes", "no"])
            if  answer == "yes":
                answers.append(1)
            else:
                answers.append(0)
        
        # Add a submit button
        submit_button = st.form_submit_button(label="Submit Answers")

        if submit_button:
            final_csv_data = [questions, answers]
            print(final_csv_data)
            write_to_csv(file_name = the_file_name, csv_data = final_csv_data)
        

if __name__ == "__main__":
    main(the_file_name = "main_V2_data.csv")