import os
import sys
sys.path.append('../../')
sys.path.append('../')
sys.path.append('../code/')

import numpy as np
import pandas as pd

import lime
from lime.lime_text import LimeTextExplainer

import sklearn
import sklearn.ensemble
import sklearn.metrics
import sklearn.feature_extraction
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline



import nltk
from nltk.corpus import stopwords
from helper.utils import train_test_split_df


# Download the stopwords from nltk
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))


# Function to remove stopwords from a sentence
def remove_stopwords(sentence):
    words = sentence.split()
    filtered_sentence = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)

def auto_trigWords(labels, version = 3, num_top_words = 2):

    ## remove this, 3 is because, as test and val are from the manual made scenarios
    # if version == 2:
    #     version = 3

    print(version, "3")

    processed_data_path = "../../data/processed/version"
    train_path = processed_data_path + str(version) + "/train.csv"
    val_path = processed_data_path + str(version) + "/val.csv"
    test_path = processed_data_path + str(version) + "/test.csv"

    full_path = processed_data_path + str(version) + "/full.csv"

    if os.path.exists(train_path):
        df_train, df_val, df_test = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)
    else:
        df_full =  pd.read_csv(full_path)
        df_train, df_val, df_test = train_test_split_df(df_full, labels)

    X_train  = df_train["Scenario"]
    y_train = df_train[labels]

    df_val_test = pd.concat([df_val,df_test], ignore_index=True)

    X_test  = df_val_test["Scenario"]
    y_test = df_val_test[labels]

    class_names = sorted(y_train.value_counts().index.tolist())


    # Apply the function to the X_test array
    X_train= np.array([remove_stopwords(sentence) for sentence in X_train])
    X_test = np.array([remove_stopwords(sentence) for sentence in X_test])

    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)


    nb = MultinomialNB(alpha=.01)
    nb.fit(train_vectors, y_train)


    c = make_pipeline(vectorizer, nb)

    explainer = LimeTextExplainer(class_names=class_names)


    label_instances = class_names
    label_instances.sort()


    # Creating the label map
    label_map = {label: idx for idx, label in enumerate(label_instances)}
    print(label_map)

    # Creating the reverse label map
    reverse_label_map = {idx: label for idx, label in enumerate(label_instances)}


    labels_instances_map = [label_map[item] for item in label_instances]


    trigWords = [set() for _ in range(len(label_instances))]


    for idx in range(len(X_test)):
        exp = explainer.explain_instance(X_test[idx], c.predict_proba, num_features = num_top_words, labels=labels_instances_map)
        pred = nb.predict(test_vectors[idx]).reshape(1, -1)[0, 0]


        words_score = exp.as_list()
        words = {item[0].lower() for item in words_score}

        for label_instance in label_instances:
            if pred == label_instance:
                trigWords[label_map[pred]] = trigWords[label_map[pred]] .union(words)
    return trigWords
