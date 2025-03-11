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
    { 'tag': 'greeting', 'patterns': ['Hi', 'Hello', 'Hey', 'What’s up', 'How are you'], 'responses': ['Hi there!', 'Hello!', 'Hey!', 'Nothing much.', 'I’m fine, thank you.'] },
    { 'tag': 'goodbye', 'patterns': ['Bye', 'See you later', 'Goodbye', 'Take care'], 'responses': ['Goodbye!', 'See you later!', 'Take care!'] },
    { 'tag': 'thanks', 'patterns': ['Thank you', 'Thanks', 'Thanks a lot', 'I appreciate it'], 'responses': ["You're welcome!", "No problem!", "Glad I could help."] },
    { 'tag': 'about', 'patterns': ['What can you do?', 'Who are you?', 'What are you?', 'What is your purpose?'], 'responses': ['I am a chatbot!', 'My purpose is to assist you.', 'I can answer questions and provide help.'] },
    { 'tag': 'name', 'patterns': ['What is your name?', 'Who are you called?', 'Do you have a name?'], 'responses': ['I am just a chatbot.', 'You can call me ChatBot.'] },
    { 'tag': 'joke', 'patterns': ['Tell me a joke', 'Make me laugh', 'Say something funny'], 'responses': ['Why did the scarecrow win an award? Because he was outstanding in his field!', 'Why don’t skeletons fight each other? They don’t have the guts.'] },
    { 'tag': 'weather', 'patterns': ['What’s the weather like?', 'Is it raining today?', 'Will it be sunny tomorrow?'], 'responses': ['I’m not sure, but you can check a weather app!', 'It depends on your location.'] },
    { 'tag': 'time', 'patterns': ['What time is it?', 'Can you tell me the time?', 'Clock time?'], 'responses': ['I don’t have a watch, but you can check your phone!'] },
    { 'tag': 'food', 'patterns': ['What should I eat?', 'Suggest me some food', 'I am hungry'], 'responses': ['Pizza sounds good!', 'How about a burger?', 'Maybe try some pasta!'] },
    { 'tag': 'mood', 'patterns': ['How are you feeling?', 'Are you happy?', 'Are you sad?'], 'responses': ['I’m always here to help!', 'I don’t have emotions, but I’m happy to chat with you.'] },
    { 'tag': 'age', 'patterns': ['How old are you?', 'What is your age?', 'When were you created?'], 'responses': ['I am timeless.', 'I exist in the digital world, so I have no age.'] },
    { 'tag': 'creator', 'patterns': ['Who created you?', 'Who is your developer?', 'Who made you?'], 'responses': ['I was created by a developer like you!', 'A smart human made me.'] },
    { 'tag': 'love', 'patterns': ['Do you love me?', 'Can you fall in love?', 'Do you have feelings?'], 'responses': ['I don’t have emotions, but I appreciate our conversation!', 'I’m just a chatbot, but I’m here to chat with you!'] },
    { 'tag': 'help', 'patterns': ['Can you help me?', 'I need assistance', 'Help me with something'], 'responses': ['Of course! What do you need help with?', 'Sure! Let me know what you need assistance with.'] },
    { 'tag': 'hobby', 'patterns': ['What do you do for fun?', 'What are your hobbies?', 'Do you like anything?'], 'responses': ['I like chatting with people like you!', 'Talking to you is my favorite activity.'] },
    { 'tag': 'movie', 'patterns': ['Recommend me a movie', 'What should I watch?', 'Suggest a good film'], 'responses': ['You could watch Inception!', 'How about The Matrix?', 'Try Interstellar!'] },
    { 'tag': 'music', 'patterns': ['Suggest a song', 'Recommend me some music', 'What’s a good song?'], 'responses': ['How about some classic rock?', 'Try listening to pop hits!', 'Maybe some lo-fi beats would be nice.'] },
    { 'tag': 'sports', 'patterns': ['What is your favorite sport?', 'Do you watch sports?', 'Tell me about sports'], 'responses': ['I don’t play sports, but I can talk about them!', 'Football is very popular!', 'Basketball is exciting!'] },
    { 'tag': 'travel', 'patterns': ['Suggest me a travel destination', 'Where should I go for vacation?', 'Best places to visit?'], 'responses': ['Paris is beautiful!', 'You could visit Tokyo.', 'How about a trip to Bali?'] },
    { 'tag': 'math', 'patterns': ['Solve 5+3', 'What is 10*2?', 'Can you do math?'], 'responses': ['I can do basic math! Ask me a question.', 'Sure! What calculation do you need?'] },
    { 'tag': 'programming', 'patterns': ['What is Python?', 'How do I learn programming?', 'Best programming language?'], 'responses': ['Python is a great language!', 'You can learn programming online!', 'Try practicing coding every day!'] },
    { 'tag': 'news', 'patterns': ['Tell me the latest news', 'What’s happening in the world?', 'Any news updates?'], 'responses': ['You can check news websites for the latest updates.', 'I don’t have live news, but you can check Google News.'] },
    { 'tag': 'study', 'patterns': ['How can I study better?', 'Give me study tips', 'Best way to focus on studying?'], 'responses': ['Try making a schedule!', 'Take breaks while studying.', 'Use flashcards to remember key points.'] },
    { 'tag': 'exercise', 'patterns': ['How can I get fit?', 'Recommend me a workout', 'Best exercise to stay healthy?'], 'responses': ['Try jogging every morning.', 'Yoga is great for flexibility.', 'Weight training helps build strength.'] },
    { 'tag': 'health', 'patterns': ['How to stay healthy?', 'Give me health tips', 'Best way to boost immunity?'], 'responses': ['Eat healthy food and exercise!', 'Drink enough water daily.', 'Get enough sleep every night.'] },
    { 'tag': 'motivation', 'patterns': ['Give me motivation', 'Motivate me', 'Inspire me'], 'responses': ['Believe in yourself!', 'You can do anything you set your mind to.', 'Every day is a new opportunity!'] },
    { 'tag': 'sleep', 'patterns': ['How much sleep do I need?', 'Best way to sleep better?', 'I can’t sleep, any tips?'], 'responses': ['Try to sleep at least 7-8 hours.', 'Avoid screens before bedtime.', 'Listen to calming music to fall asleep.'] },
    { 'tag': 'pets', 'patterns': ['Do you like pets?', 'Tell me about cats', 'Should I get a dog?'], 'responses': ['Pets are amazing companions!', 'Dogs are loyal and friendly.', 'Cats are independent but loving.'] },
    { 'tag': 'space', 'patterns': ['Tell me about space', 'Is there life on other planets?', 'What’s in the universe?'], 'responses': ['Space is vast and full of mysteries!', 'Scientists are exploring the universe.', 'There may be other planets with life!'] },
    { 'tag': 'history', 'patterns': ['Tell me about history', 'Who was the first president?', 'What happened in the past?'], 'responses': ['History is full of amazing events!', 'The first U.S. president was George Washington.', 'You can read history books for more details.'] }
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
