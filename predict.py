#Text Data Preprocessing Lib
import nltk

import json
import pickle
import numpy as np
import random

ignore_words = ['?', '!',',','.', "'s", "'m"]

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

model = tensorflow.keras.models.load_model("chatbot_model.h5")

words = pickle.load(open("words.pkl", 'rb'))
classes = pickle.load(open("classes.pkl", 'rb'))
intense = json.loads(open("intents.json").read())

def preprocess_user_input(user_input):
    input_1 = nltk.word_tokenize(user_input)
    input_2 = get_stem_words(input_1, ignore_words)
    input_2 = sorted(list(set(input_2)))

    bag = []
    bag_of_word  = []
    for i in words:
        if i in input_2:
            bag_of_word.append(1)
        else:
            bag_of_word.append(0)
    bag.append(bag_of_word)
    return np.array(bag)


def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)
    prediction_class_label = np.argmax(prediction[0])
    return prediction_class_label

def bot_response(user_input):
    predicted_class_label = bot_class_prediction(user_input)
    predicted_class = classes[predicted_class_label]
    for h in 






