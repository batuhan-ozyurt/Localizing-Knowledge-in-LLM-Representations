from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from memformers.models.membart import MemBartForConditionalGeneration

import os
import nltk
import pandas as pd
import torch
import numpy as np
from jinja2 import Template
import pickle
from collections import defaultdict

from fuzzywuzzy import fuzz

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
from math import log

import pdb

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

all_annotations = {}

for item in dir_list_annotations:
    
    f = open(os.path.join(path_annotations, item), 'r')
    annotations = pd.read_csv(f, sep="\t")
    annotations = annotations.values
    f.close()
    
    all_annotations[item] = annotations
    
story_info = {}
for story in all_annotations:
    story_info[story] = {}
    for character_list in all_annotations[story]:
        character_name = character_list[1]
        story_info[story].setdefault(character_name, [])
        story_info[story][character_name].append((character_list[0], character_list[2], character_list[3]))
        story_info[story][character_name].sort(key=lambda x: x[0])
        
model = MemBartForConditionalGeneration.from_pretrained("qywu/membart-large")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

memory_enhanced_instance = ModelInstance(model, tokenizer, device)

class MemoryEnhancedResults():
    def __init__(self, prompt, out, ind, character, gold_locations, story_no, k, query_point, exact_match, fuzzy_match):
        self.prompt = prompt
        self.out = out
        self.ind = ind
        self.character = character
        self.gold_locations = gold_locations
        self.story_no = story_no
        self.k = k
        self.query_point = query_point
        self.exact_match = exact_match
        self.fuzzy_match = fuzzy_match

def get_memory_states(memory_enhanced_instance, all_prompt_drafts):
    
    model = memory_enhanced_instance.model
    tokenizer = memory_enhanced_instance.tokenizer
    device = memory_enhanced_instance.device
    
    all_memory_states = []
    memory_states = model.construct_memory(batch_size=1)
    all_memory_states.append(memory_states)
    
    for prompt_draft in all_prompt_drafts:
    
        input_ids = torch.LongTensor([tokenizer.encode(prompt_draft)]).to(device)
        encoder_outputs = model.model.encoder(input_ids=input_ids, memory_states=memory_states, attention_mask=None)
        memory_states = encoder_outputs.memory_states
        all_memory_states.append(memory_states)
        
    return all_memory_states       

def predictor(memory_enhanced_instance):

    model = memory_enhanced_instance.model
    tokenizer = memory_enhanced_instance.tokenizer
    device = memory_enhanced_instance.device
    
    all_result_objects = defaultdict(list)

    for item in dir_list_annotations:

        print(item)

        story_no = item[len("Andersen_story"):-len(".txt")]

        f = open(os.path.join(path_andersen, item), 'r') 
        story = f.read()
        f.close()

        characters = story_info[item].keys()
        story = remove_new_lines(story)

        big_prompt = "Story:\n" + story
        big_prompt_tokens = tokenizer.encode(big_prompt)
        toks_per_segment = 985
        no_segments = len(big_prompt_tokens) // toks_per_segment

        all_tokens = [ big_prompt_tokens[toks_per_segment*i : toks_per_segment*(i+1)] for i in range(no_segments) ]

        all_tokens.append(big_prompt_tokens[toks_per_segment*(no_segments): ])

        all_prompt_drafts = [ tokenizer.decode(x, skip_special_tokens=True) for x in all_tokens ]
        
        all_memory_states = get_memory_states(memory_enhanced_instance, all_prompt_drafts)

        x = 0
        story_lens = []
        for prompt_draft in all_prompt_drafts:
            if prompt_draft[:len("Story:\n")] == "Story:\n":
                x += len(prompt_draft) - len("Story:\n")
            else:
                x += len(prompt_draft)
            story_lens.append(x)

        for i, prompt_draft in enumerate(all_prompt_drafts):
            
            print(i)
            
            for k in range(1, num_templates+1):

                for character in characters:

                    #print(character)

                    tuples = story_info[item][character]

                    grammatical_number = tuples[0][2]

                    if i != 0:
                        pos = story_lens[i-1]
                    else:
                        pos = 0

                    prompt, pos_last = create_membart_prompt(tokenizer, k, prompt_draft, character, grammatical_number, 1009)
                    pos += pos_last

                    if prompt[-1] == " ":
                        prompt += "<mask>"
                    else:
                        prompt += " <mask>"

                    check = False

                    for num_tupl, tupl in enumerate(tuples):

                        if pos < tupl[0]:
                            check = True
                            last_tupl = num_tupl
                            break
                        ind = tupl[0]
                        gold_locations = tupl[1]

                    if check:
                        input_ids = torch.LongTensor([tokenizer.encode(prompt)]).to(device)
                        encoder_outputs = model.model.encoder(input_ids=input_ids, memory_states=all_memory_states[i], attention_mask=None)
                        #memory_states = encoder_outputs.memory_states

                        with torch.no_grad():
                            outputs = model.generate(
                                encoder_outputs=encoder_outputs,
                                decoder_start_token_id=tokenizer.bos_token_id,
                                max_length=1024,
                                num_beams=3,
                                return_dict_in_generate=True,
                            )

                        out = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

                        match1, match2 = exactly_or_fuzzily_matched(out, character, gold_locations)
                        result_object =  MemoryEnhancedResults(prompt, out, ind, character, gold_locations, story_no, k, pos, match1, match2)
                        all_result_objects[k].append(result_object)                 

    return all_result_objects

all_result_objects = predictor(memory_enhanced_instance)

with open("membart_predictions.txt", "wb") as f:
    pickle.dump(all_result_objects, f)
    
