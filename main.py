import csv

def ask_question(prompt):
    while True:
        answer = input(f"{prompt} (yes/no): ").strip().lower()
        if answer in ["yes", "no"]:
            if answer == "yes":
                return 1
            return 0
        print("Please answer with 'yes' or 'no'.")

def get_dict():
    
    responses = {
        # Diabetes-related
        "age_above_45": ask_question("Is your age above 45?"),
        "family_history_diabetes": ask_question("Do you have a family history of diabetes?"),
        "obese_or_high_bmi": ask_question("Do you have obesity or a high BMI?"),
        "hypertension": ask_question("Do you have a history of hypertension?"),
        "low_physical_activity": ask_question("Do you have low physical activity levels?"),
        "high_sugar_diet": ask_question("Do you consume high sugar/carbohydrate diets?"),
        "smoke_or_alcohol": ask_question("Do you smoke or consume alcohol regularly?"),

        # Pneumonia-related
        "recent_respiratory_infection": ask_question("Have you had any recent respiratory infections (e.g. cold, flu)?"),
        "asthma_or_copd": ask_question("Do you have asthma or COPD history?"),
        "immunocompromised": ask_question("Are you immunocompromised (e.g. HIV, steroids, chemo)?"),
        "smoking_history": ask_question("Do you have a history of smoking?"),
        "age_extremes": ask_question("Are you a young child or above 65 years old?"),
        "chronic_illnesses": ask_question("Do you have any chronic illnesses (esp. heart or lung diseases)?"),
        "vaccination_status": ask_question("Have you received pneumococcal or flu vaccinations?"),

        # TB-related
        "contact_tb_patients": ask_question("Have you had close contact with TB patients?"),
        "hiv_positive": ask_question("Are you HIV positive?"),
        "travel_to_tb_areas": ask_question("Have you traveled or lived in high TB prevalence regions?"),
        "immune_suppression": ask_question("Are you undergoing any immune-suppressing treatments (e.g. cancer therapy)?"),
        "malnutrition": ask_question("Are you affected by malnutrition?"),
        "other_chronic_conditions": ask_question("Do you have chronic conditions (like diabetes)?")
    }

    return responses

def convert_to_csv_format():
    dict_responses = get_dict()
    list_responses = []
    keys = dict_responses.keys()
    values = dict_responses.values()
    

def write_to_csv(file_name):
    with open(file_name) as file1:
        file1.write()