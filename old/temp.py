import csv
list1 = [["Do you have a persistent cough lasting more than 2 weeks?",
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
                "Are you above 45 years of age?"],[]
                
            ]

with open("file1.csv", 'w', newline='\n') as file1:
    writer = csv.writer(file1)
    for i in list1:
        writer.writerow(list(i))