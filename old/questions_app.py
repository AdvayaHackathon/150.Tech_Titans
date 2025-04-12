import streamlit as st

# Title of the app
st.title("Question and Answer Collector")

# Define a list of questions
questions = [
    "What is your favorite color?",
    "What is your favorite animal?",
    "What is your favorite food?"
]

# Initialize a list to store answers
answers = []

# Create a form to collect all answers at once
with st.form(key="question_form"):
    # Loop through each question and create a text input
    for i, question in enumerate(questions, 1):
        answer = st.text_input(f"Question {i}: {question}", key=f"q{i}")
        answers.append(answer)
    
    # Add a submit button
    submit_button = st.form_submit_button(label="Submit Answers")

# Display the answers after submission
if submit_button:
    st.write("### Your Answers:")
    for i, (question, answer) in enumerate(zip(questions, answers), 1):
        if answer.strip():  # Only show non-empty answers
            st.write(f"**Question {i}:** {question}**Answer:** {answer}")
        else:
            st.write(f"**Question {i}:** {question}**Answer:** (No response provided)")
    
    # Return the list of answers (for debugging or further use)
    st.write("### List of Answers:")
    st.write(answers)
