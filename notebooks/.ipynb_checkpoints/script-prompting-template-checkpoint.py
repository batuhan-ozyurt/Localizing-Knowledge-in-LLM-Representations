#TODO: Add here your imports

from transformers import AutoTokenizer, BartForConditionalGeneration

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
sys.path.append('/scratch/users/bozyurt20/hpc_run/')
sys.path.append('/scratch/users/bozyurt20/hpc_run/text_injection')
from text_injection.util_research import *

## TODO: Add the number of template options for prompting

num_templates = 23

path_andersen = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Andersen"
path_fanny = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Fanny Fern"
path_annotations = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Annotations"

dir_list_andersen = os.listdir(path_andersen)
dir_list_fanny = os.listdir(path_fanny)
dir_list_annotations = os.listdir(path_annotations)

stop_words = set(stopwords.words("english"))

class ModelInstance():
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.model = self.model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
class ResultObject():
    def __init__(self, prompt, out, story_no, k, match1, match2):
        self.prompt = prompt
        self.out = out
        self.story_no = story_no
        self.k = k
        self.match1 = match1
        self.match2 = match2
        
## TODO: Add Your model and tokenizer

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_instance = ModelInstance(model, tokenizer, device)

## TODO: example pipeline

from transformers import AutoTokenizer, BartForConditionalGeneration

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

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
            
            character = line[1]
            gold_answer = line[2]
            grammatical_number = line[3]

            gold_locations = gold_answer.split("/")
            
            for k in range(1, 24):
                
                y = line[0]
                x = y - 5120

                if x < len_title:
                    text = story[len_title:y]

                else:
                    x = story[x:y].find(" ") + x
                    text = story[x:y]                
                
                text = text_clean_ending(text)
                text = remove_new_lines(text)                    
                
                ## TODO: Write the predictor lines
                
                prompt, context2 = create_prompt_clipped(tokenizer, k, text, character, grammatical_number, 1024)
                
                inputs = tokenizer([prompt], max_length=1024, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                
                with torch.no_grad():
                    summary_ids = model.generate(input_ids, min_length=0, max_length=20)
                    
                out = tokenizer.decode(summary_ids[0], skip_special_tokens=True)                
                
                ## Ends here
                
                match1, match2 = exactly_or_fuzzily_matched(out, character, gold_locations)
                result_object = ResultObject(prompt, out, story_no, k, match1, match2)
                all_result_objects[k].append(result_object)
                
    return all_result_objects

all_result_objects = predictor(model_instance)

## TODO: define a path name for saving the results

with open("bart_large_predictions.txt", "wb") as f:
    pickle.dump(all_result_objects, f)
    
