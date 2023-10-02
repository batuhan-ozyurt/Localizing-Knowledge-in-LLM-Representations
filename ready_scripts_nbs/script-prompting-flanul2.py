from transformers import T5ForConditionalGeneration, AutoTokenizer

import os
import nltk
import pandas as pd
import torch
import numpy as np
from jinja2 import Template
import xmltodict
import pickle
from collections import defaultdict

from fuzzywuzzy import fuzz

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import sys
sys.path.append('/scratch/users/bozyurt20/hpc_run/utilities')
from util_research import *

num_templates = 23

path_andersen = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Andersen"
path_fanny = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Fanny Fern"
path_annotations = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Annotations"

dir_list_andersen = os.listdir(path_andersen)
dir_list_fanny = os.listdir(path_fanny)
dir_list_annotations = os.listdir(path_annotations)

stop_words = set(stopwords.words("english"))
       
model = T5ForConditionalGeneration.from_pretrained("google/flan-ul2", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_instance = ModelInstance(model, tokenizer, device)

"""input_string = "Answer the following question by reasoning step by step. The cafeteria had 23 apples. If they used 20 for lunch, and bought 6 more, how many apples do they have?"                                               

inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(inputs, max_length=200)

print(tokenizer.decode(outputs[0]))
# <pad> They have 23 - 20 = 3 apples left. They have 3 + 6 = 9 apples. Therefore, the answer is 9.</s>
"""

def predictor(model_instance):
    
    model = model_instance.model
    tokenizer = model_instance.tokenizer
    device = model_instance.device
    
    all_result_objects = defaultdict(list)
    
    for item in dir_list_annotations:

        print(item)

        story_no = item[len("Andersen_story"):-len(".txt")]

        f = open(os.path.join(path_andersen, item), 'r') 
        story = f.read()
        f.close()
        
        paragraphs = story.split("\n\n")
        paragraph = paragraphs[0]
        len_title = len(paragraph) + 2   
        
        annotations = all_annotations[item]
        
        for line in annotations:
            
            ind = line[0]
            character = line[1]
            gold_answer = line[2]
            grammatical_number = line[3]

            gold_locations = gold_answer.split("/")
            
            for k in range(1, num_templates+1):
                
                y = line[0]
                x = y - 5120

                if x < len_title:
                    text = story[len_title:y]

                else:
                    x = story[x:y].find(" ") + x
                    text = story[x:y]                
                
                text = text_clean_ending(text)
                text = remove_new_lines(text)                    
                
                prompt, context2 = create_prompt_clipped(tokenizer, k, text, character, grammatical_number, 512)
                
                inputs = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(inputs, max_new_tokens=30)

                out = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                match1, match2 = exactly_or_fuzzily_matched(out, character, gold_locations)
                result_object = ResultObject(prompt, out, ind, character, gold_locations, story_no, k, match1, match2)
                all_result_objects[k].append(result_object)
                
    return all_result_objects

all_result_objects = predictor(model_instance)

with open("flan_ul2_predictions.txt", "wb") as f:
    pickle.dump(all_result_objects, f)
    
