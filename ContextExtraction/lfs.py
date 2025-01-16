import numpy as np
import re
import enum

import os
import pickle


import sys
sys.path.append('../../')
sys.path.append('../')

from spear.labeling import labeling_function, LFSet, ABSTAIN, preprocessor
from helper.con_scorer import word_similarity
from LIME import auto_trigWords




# Define trigger words for each class
# trigger_words = [{"mission"}, {"maintenance"}]



# Preprocessor function to convert text to lowercase and strip whitespace
@preprocessor()
def convert_to_lower(x):
    return x.lower().strip()


# Add manual labeling functions
def manual_trigWords(words):

    trigWord1 = {words}
    trigWord2 = set(words.replace(","," ").split())

    trigWord = trigWord1.union(trigWord2)
    trigWord = {element.lower() for element in trigWord}
    return trigWord

# note: add auto labeling functions here

# Function to create labeling functions, create some other labeling functions here
def create_labeling_function(lf_name, label, trig_words, cont_scorer=None, THRESHOLD = 0.8):
    if cont_scorer:
        @labeling_function(name=lf_name, cont_scorer=cont_scorer, resources=dict(keywords=trig_words), pre=[convert_to_lower], label=label)
        def lf(c, **kwargs):
            print("trig_words-->>",trig_words,kwargs["continuous_score"])
            if kwargs["continuous_score"] >= THRESHOLD:
                return label
            else:
                return ABSTAIN
    else:
        @labeling_function(name=lf_name, resources=dict(keywords=trig_words), pre=[convert_to_lower], label=label)
        def lf(x, **kwargs):
            print("trig_words-->>",trig_words,kwargs["continuous_score"])
            if len(kwargs["keywords"].intersection(x.split())) > 0:
                return label
            else:
                return ABSTAIN
    return lf



def design_lf(labels,label_instances,version,num_top_words = 2):
    # Dynamically create ClassLabels enum
    ClassLabels = enum.Enum('ClassLabels', {label_instance: idx for idx, label_instance in enumerate(label_instances)})

        # Initialize the LFSet with the labels
    rules = LFSet(ClassLabels)
    lfs_list = []

    # Loop to create and add labeling functions
    i = 1
    trigger_words_manual = [manual_trigWords(label_instance) for label_instance in label_instances]

    ## instead of running trigwords agaian and again, please save them
   

    auto_lf_path = f"../checkpoint/auto_trigWords/version{version}/{labels}.pkl"
    os.makedirs(os.path.dirname(auto_lf_path), exist_ok=True)

    if  os.path.exists(auto_lf_path):
        with open(auto_lf_path, 'rb') as f:
            trigWords = pickle.load(f)

    else:
        trigWords = auto_trigWords(labels = labels, version = version, num_top_words = num_top_words)
        with open(auto_lf_path , 'wb') as f:
            pickle.dump(trigWords , f)


    auto1 =  [set(list(s)[:len(s)//2]) for s in trigWords]
    auto2 = [set(list(s)[len(s)//2:]) for s in trigWords]
    ## note : combine these two and select the tops
    trigger_words_list = [trigger_words_manual, auto1, auto2] 


    for trigger_words in trigger_words_list:
        # print(trigger_words)
        for (label_instance, trig_words) in zip(label_instances, trigger_words):
            # Check if the current trigger words are in the manual list/set of trigger words
           
            if trigger_words == trigger_words_manual:
                lf_name = f"manual_{i}"
                clf_name = f"manual_{i+1}"
            else:
                lf_name = f"auto_{i}"
                clf_name = f"auto_{i+1}"

            label = ClassLabels[label_instance]
            print("##################")
            print(label)
            print(trig_words)
            print("#################")

            lf = create_labeling_function(lf_name, label, trig_words)
            clf = create_labeling_function(clf_name, label, trig_words, cont_scorer=word_similarity)

            lfs_list.append(lf)
            lfs_list.append(clf)

            # Increment `i` by 2 for the next pair of labeling functions
            i += 2

    # Add the labeling functions to the rules set
    rules.add_lf_list(lfs_list)

    return ClassLabels, rules 