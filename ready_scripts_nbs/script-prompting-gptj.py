from transformers import GPTJForCausalLM, AutoTokenizer

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
        
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_instance = ModelInstance(model, tokenizer, device)

"""prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

gen_tokens = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]"""

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
                
                prompt, context2 = create_prompt_clipped(tokenizer, k, text, character, grammatical_number, 2048)
                
                encoded = tokenizer(prompt, return_tensors="pt")
                
                input_ids = encoded.input_ids.to(device)
                attn_mask = encoded.attention_mask.to(device)
                
                with torch.no_grad():
                    gen_tokens = model.generate(
                        input_ids,
                        do_sample=False,
                        attention_mask=attn_mask,
                        #temperature=0.9,
                        max_new_tokens=40,
                    )
                out = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
                
                match1, match2 = exactly_or_fuzzily_matched(out, character, gold_locations)
                result_object = ResultObject(prompt, out, ind, character, gold_locations, story_no, k, match1, match2)
                all_result_objects[k].append(result_object)
                
    return all_result_objects

all_result_objects = predictor(model_instance)

with open("gptj_predictions.txt", "wb") as f:
    pickle.dump(all_result_objects, f)
    
