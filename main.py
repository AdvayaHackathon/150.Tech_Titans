import csv

def ask_question(prompt):
    while True:
        answer = input(f"{prompt} (yes/no): ").strip().lower()
        if answer in ["yes", "no"]:
            if answer == "yes":
                return 1
            return 0
        print("Please answer with 'yes' or 'no'.")


def get_dict(disease_questions):
    dict_questions_answers = {}
    for i in disease_questions:
        disease = disease_questions[i]
        for type in disease:
            for question in disease[type]:
                dict_questions_answers[question] = ask_question(question)
    return dict_questions_answers

def convert_to_csv_format(disease_questions):
    dict_responses = get_dict(disease_questions)
    list_responses = []
    keys = list(dict_responses.keys())
    values = list(dict_responses.values())
    print(keys, values)
    list_responses.append(keys)
    list_responses.append(values)
    return list_responses

def write_to_csv(file_name, csv_data):
    with open(file_name, 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerows(csv_data)
def count_lines(disease_questions):
    dict_questions_answers = {}
    for i in disease_questions:
        number = 0
        disease = disease_questions[i]
        for type in disease:
            for question in disease[type]:
                number += 1
        dict_questions_answers[i] = number
    return dict_questions_answers

def put_into_3_files(count, read, file_name1, file_name2, file_name3):
    check1 = count['tuberculosis']
    check2 = check1 + count['pneumonia']
    check3 = check2 + count['lung_cancer']

    csv_data1 = [read[0][:check1], read[1][:check1]]
    csv_data2 = [read[0][check1:check2], read[1][check1:check2]]
    csv_data3 = [read[0][check2:check3], read[1][check2:check3]]

    with open(file_name1, 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerows(csv_data1)

    with open(file_name2, 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerows(csv_data2)

    with open(file_name3, 'w', newline='') as file1:
        writer = csv.writer(file1)
        writer.writerows(csv_data3)

def main():

    disease_questions = {
        "tuberculosis": {
            "symptoms": [
                "Do you have a persistent cough lasting more than 2 weeks?",
                "Are you coughing up blood or blood-stained sputum?",
                "Do you experience night sweats regularly?",
                "Have you had an unexplained weight loss recently?",
                "Do you have a prolonged fever or fatigue?"
            ],
            "medical_history": [
                "Have you been in close contact with a person who has TB?",
                "Have you previously been treated for TB?",
                "Have you traveled or lived in a region with high TB prevalence?",
                "Are you HIV positive?",
                "Are you undergoing immune-suppressive treatment (e.g. cancer therapy)?",
                "Are you malnourished or underweight?",
                "Do you have chronic illnesses such as diabetes?"
            ]
        },

        "pneumonia": {
            "symptoms": [
                "Do you have a cough with mucus or phlegm?",
                "Are you experiencing chest pain when breathing or coughing?",
                "Do you have a high fever, chills, or sweating?",
                "Are you experiencing shortness of breath?",
                "Do you feel unusually fatigued or weak?"
            ],
            "medical_history": [
                "Have you recently had a respiratory infection (like flu or cold)?",
                "Do you have asthma or COPD?",
                "Are you immunocompromised (e.g. HIV, chemo, steroids)?",
                "Do you smoke or have a smoking history?",
                "Are you a child below 5 or an adult over 65?",
                "Do you have any chronic heart or lung diseases?",
                "Have you received pneumococcal or flu vaccinations?"
            ]
        },

        "lung_cancer": {
            "symptoms": [
                "Do you have a persistent cough that doesn't go away?",
                "Are you coughing up blood or rust-colored sputum?",
                "Do you experience hoarseness or wheezing?",
                "Do you have chest pain that worsens with deep breaths?",
                "Have you experienced significant weight loss or fatigue recently?"
            ],
            "medical_history": [
                "Do you currently smoke or have a history of smoking?",
                "Are you frequently exposed to secondhand smoke, asbestos, or radon?",
                "Do you have a family history of lung cancer?",
                "Have you been exposed to occupational hazards (e.g. mining, construction)?",
                "Have you previously had radiation therapy to the chest?",
                "Do you have any chronic lung diseases (e.g. COPD, TB scars)?"
            ]
        }
    }

    count = count_lines(disease_questions)
    print(f"Lines are: {count}")
    """
    input_file = convert_to_csv_format(disease_questions)
    write_to_csv(file_name = "main_file.csv", csv_data = input_file)"""
    with open("main_file.csv", 'r', newline='') as file1:
        reader = csv.reader(file1)
        read_data = list(reader)

        put_into_3_files(count, read_data, file_name1 = "tuberculosis.csv", file_name2="pneumonia.csv", file_name3="lung_cancer.csv")



main()