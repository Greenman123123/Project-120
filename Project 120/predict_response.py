# Text Data Preprocessing Lib
import nltk
nltk.download("punkt")
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np
import random


ignore_words = ['?', '!',',','.', "'s", "'m"]

import tensorflow 
model = tensorflow.keras.models.load_model("chatbot_model.h5")
from data_preprocessing import get_stem_words

intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl","rb"))
classes = pickle.load(open("classes.pkl","rb"))

def botresponse(userInput):
    input1 = nltk.word_tokenize(userInput)
    input2 = get_stem_words(input1,ignore_words)
    input3 = sorted(list(set(input2)))
    bag = []
    bag_of_words = []
    for word in words:
        if word in input3:
            bag_of_words.append(1)
        else:
            bag_of_words.append(0)
    bag.append(bag_of_words)
    modelInput = np.array(bag)
    prediction = model.predict(modelInput)
    predictedLabel = np.argmax(prediction[0])
    predictedClass = classes[predictedLabel]
    for x in intents["intents"]:
        if x["tag"] == predictedClass:
            bot_response = random.choice(x["responses"])
            return bot_response

print("Hi I am Stella, how can I help you?")
while True:
    userInput=input("Type your message here: ")
    response = botresponse(userInput)
    print(response)