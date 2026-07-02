""" 
aichatbot.py
"""
import numpy as np
import tensorflow as tf
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Initialize the stemmer
stemmer = LancasterStemmer()

# Download nltk dependencies
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import string
punct_dict = dict((ord(punct), None) for punct in string.punctuation)

# Load and prepare data
categories = []
questions = []
answers = []

# read in file aichatbot.txt
with open("aichatbot.txt", "r") as f:
    while True:
        line = f.readline().strip()
        if not line:
            break
        categories.append(line)
        questions.append(f.readline().lower().strip())
        answers.append(f.readline().lower().strip())

# Tokenize and remove stop words
word_tokens_stop = []
questions_tokenized_stopped = []
for i, question in enumerate(questions):
    question = question.translate(punct_dict)
    tokens = nltk.word_tokenize(question)
    tokens_stop = [w for w in tokens if not w in stop_words]
    word_tokens_stop.extend(tokens_stop)
    questions_tokenized_stopped.append(tokens_stop)

# Stem words
stemmed_words = [stemmer.stem(w) for w in word_tokens_stop]
stemmed_words = sorted(list(set(stemmed_words)))

sorted_categories = sorted(list(set(categories)))

# Prepare training data
training = []
output = []

# process questions
for i, question in enumerate(questions_tokenized_stopped):
    training_row = []
    stemmed_question = [stemmer.stem(token) for token in question]

    for w in stemmed_words:
        training_row.append(1 if w in stemmed_question else 0)

    output_row = [0] * len(sorted_categories)
    output_row[sorted_categories.index(categories[i])] = 1

    training.append(training_row)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Build the neural network using TensorFlow 2.x
input_size = len(training[0])
output_size = len(output[0])

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_size,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(training, output, epochs=1000, batch_size=8)

# Function to process input and predict category
def get_response(query, words):
    row = [0] * len(words)
    query = query.lower().translate(punct_dict)
    tokens = nltk.word_tokenize(query)
    tokens_stop = [w for w in tokens if w not in stop_words]
    stemmed_tokens = [stemmer.stem(word) for word in tokens_stop]

    for stemmed_word in stemmed_tokens:
        for i, w in enumerate(words):
            if w == stemmed_word:
                row[i] = 1

    return np.array(row)

# Chat function
def chat():
    print("Hi, I am Chatbot. Ask me questions about chatbots. Type 'bye' to exit.")
    while True:
        query = input("> ")
        if query.lower() == "bye":
            print("Have a nice day!")
            break

        response = get_response(query, stemmed_words)
        
        # make prediction
        results = model.predict(np.array([response]),verbose=0)
        results_index = np.argmax(results)
        tag = sorted_categories[results_index]
        print(tag)
        
        # get random response from category
        responses = [answers[i] for i, category in enumerate(categories) if category == tag]
        print(random.choice(responses))

# Start the chatbot
chat()