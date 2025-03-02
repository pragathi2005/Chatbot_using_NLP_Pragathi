import nltk
import random
import os
import ssl
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath('nltk_data'))
nltk.download('punkt')

# Define intents
intents = [
    { 'tag':'greeting', 'patterns':['Hi','Hello','Hey','Whats up','How are you'], 'responses':['Hi there','Hello','Hey','Nothing much','I\'m fine, thank you'] },
    { 'tag':'goodbye', 'patterns':['Bye','See you later','Good bye','Take care'], 'responses':['Goodbye!','See you later','You too'] },
    { 'tag': 'thanks', 'patterns': ['Thank you', 'Thanks', 'Thanks a lot', 'I appreciate it'], 'responses': ["You're welcome", "No problem", "Glad I could help"] },
    { 'tag': 'about', 'patterns': ['What can you do', 'Who are you', 'What are you', 'What is your purpose'], 'responses': ['I am a chatbot', 'My purpose is to assist you', 'I can answer questions and provide assistance'] }
]

# Prepare training data
patterns = []
tags = []
for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Convert text to numerical format
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train SVM model
model = SVC(kernel='linear')
model.fit(X, tags)

# Chatbot response function
def chatbot_response(user_input):
    X_input = vectorizer.transform([user_input])
    tag = model.predict(X_input)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond."

# Streamlit UI
st.title("SVM Chatbot")
st.write("Type your message below:")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_input = st.text_input("You:", "", key="user_input")

if user_input:
    response = chatbot_response(user_input)
    st.session_state['messages'].append((user_input, response))

for msg in st.session_state['messages']:
    st.text(f"You: {msg[0]}")
    st.text(f"Bot: {msg[1]}")
