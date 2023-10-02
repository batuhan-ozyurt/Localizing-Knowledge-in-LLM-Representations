from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import transformers
import torch
import os
import nltk
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from jinja2 import Template
import pickle
from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import random

from transformers.generation.logits_process import LogitsProcessorList
logits_processor = LogitsProcessorList()

import sys
sys.path.append('/scratch/users/bozyurt20/hpc_run/utilities')
sys.path.append("/scratch/users/bozyurt20/hpc_run/blobs/")
sys.path.append("/scratch/users/bozyurt20/hpc_run/Inference/")
from util_research import *
from text_to_inject import chapter2_part1, chapter2_part2, chapter2_part2_part1, chapter2_part2_part2

max_len = 512
num_layers = 24
d_model = 4096

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0pp", truncation_side="right", add_prefix_space=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#prng = np.random.RandomState(1)

babi_dataset_task1 = load_dataset('babi_qa', type='en-10k', task_no='qa1')

text_to_inject = remove_new_lines(chapter2_part1 + " " + chapter2_part2)
text_to_inject_sentences = text_to_inject.split(".")

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map='balanced', load_in_8bit=True)

class BabiDataPoint():
    def __init__(self, story, question, answer, knowledge_point):
        self.story = story
        self.question = question
        self.answer = answer
        self.knowledge_point = knowledge_point
    def add_story_encodings(self, encoding):
        self.story_encoding = encoding     
    def add_story_question_encodings(self, encoding):
        self.story_question_encoding = encoding

num_ex = len(babi_dataset_task1["train"])
all_data_points = []
uniqueness = []


for i in range(num_ex):
    
    story_text = babi_dataset_task1["train"][i]["story"]["text"]
    types = babi_dataset_task1["train"][i]["story"]["type"]
    
    story = ""
    for j, sent in enumerate(story_text):
        
        if types[j] == 0:
            story += sent + " "
        elif types[j] == 1:
            
            question = sent
            answer = babi_dataset_task1["train"][i]["story"]["answer"][j]
            
            supporting_id = int(babi_dataset_task1["train"][i]["story"]["supporting_ids"][j][0])
            knowledge_point = story.rfind(story_text[supporting_id-1]) + len(story_text[supporting_id-1])
            
            data_point = BabiDataPoint(story[:-1], question, answer, knowledge_point)
            
            if data_point.story + data_point.question not in uniqueness:
                uniqueness.append(data_point.story + data_point.question)
                all_data_points.append(data_point)

                
num_ex = len(babi_dataset_task1["test"])

for i in range(num_ex):
    story_text = babi_dataset_task1["test"][i]["story"]["text"]
    types = babi_dataset_task1["test"][i]["story"]["type"]
    
    story = ""
    for j, sent in enumerate(story_text):
        if types[j] == 0:
            story += sent + " "
        elif types[j] == 1:
            question = sent
            answer = babi_dataset_task1["test"][i]["story"]["answer"][j]
            
            supporting_id = int(babi_dataset_task1["test"][i]["story"]["supporting_ids"][j][0])
            knowledge_point = story.rfind(story_text[supporting_id-1]) + len(story_text[supporting_id-1])
            
            data_point = BabiDataPoint(story[:-1], question, answer, knowledge_point)
            
            if data_point.story + data_point.question not in uniqueness:
                uniqueness.append(data_point.story + data_point.question)
                all_data_points.append(data_point)
        
smaller_dataset = random.choices(all_data_points, k=1000)
            
class TextInjectionDataPoint():
    def __init__(self, story, question, answer, prompt, token_distance):
        self.story = story
        self.question = question
        self.answer = answer
        self.prompt = prompt
        self.token_distance = token_distance
        
text_injection_data_points = []

for num_sentence in range(len(text_to_inject_sentences)):
    for a, data_point in enumerate(smaller_dataset):
        
        story = data_point.story
        question = data_point.question
        answer = data_point.answer
        knowledge_point = data_point.knowledge_point
        
        prompt = story + " " + ".".join(text_to_inject_sentences[:num_sentence+1]) + ". " + question
        
        token_distance = len(tokenizer.encode(prompt[knowledge_point:]))
        
        text_injection_data_points.append(TextInjectionDataPoint(story, question, answer, prompt, token_distance))
        
class TextInjectionBabiResultObject():
    def __init__(self, story, question, answer, prompt, token_distance, answer_predicted):
        self.story = story
        self.question = question
        self.answer = answer
        self.prompt = prompt
        self.token_distance = token_distance
        self.answer_predicted = answer_predicted
        
answer_objects = []

for data_point in text_injection_data_points:
    story = data_point.story
    question = data_point.question
    answer = data_point.answer
    prompt = data_point.prompt
    token_distance = data_point.token_distance
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=10)
    answer_predicted = tokenizer.decode(out[0], skip_special_tokens=True)
    
    answer_objects.append(TextInjectionBabiResultObject(story, question, answer, prompt, token_distance, answer_predicted))


with open("my_dataset.txt", "wb") as f:
    pickle.dump(answer_objects, f)