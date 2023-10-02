from transformers import T5Tokenizer, T5ForConditionalGeneration

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
from util_research import *

## TODO: Add the number of template options for prompting

num_templates = 23

path_andersen = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Andersen"
path_fanny = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Fanny Fern"
path_annotations = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Annotations"

dir_list_andersen = os.listdir(path_andersen)
dir_list_fanny = os.listdir(path_fanny)
dir_list_annotations = os.listdir(path_annotations)

stop_words = set(stopwords.words("english"))
        
## TODO: Add Your model and tokenizer

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl", device_map="auto", load_in_8bit=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_instance = ModelInstance(model, tokenizer, device)

## TODO: Declare the amount of tokens to be appended

tokens_to_append = [x for x in range(0, 501, 30)]

def predictor(text_injection_instance):
    
    model = text_injection_instance.model
    tokenizer = text_injection_instance.tokenizer
    device = text_injection_instance.device
    text_to_inject_tokenized = text_injection_instance.text_to_inject_tokenized
    tokens_to_append = text_injection_instance.tokens_to_append

    all_result_objects = {}
    for k in range(1, num_templates+1):
        all_result_objects[k] = []

    for item in dir_list_annotations:
        
        print(item)
        
        story_no = item[len("Andersen_story"):-len(".txt")]

        f = open(os.path.join(path_andersen, item), 'r') 
        story = f.read()
        f.close()
        
        annotations = all_annotations[item]

        paragraphs = story.split("\n\n")
        no_paragraphs = len(paragraphs)
        paragraph = paragraphs[0]
        paragraph = paragraph.replace("\n", " ")
        len_title = len(paragraph) + 2        

        i = 0 # line number in the annotation file
        j = 0 # paragraph number

        char_count = len_title   

        while i < len(annotations) and j < no_paragraphs:

            if char_count + 2 >= annotations[i][0]:

                line = annotations[i]
                ind = line[0]
                character = line[1]
                gold_answer = line[2]
                grammatical_number = line[3]
                context = paragraph
                context = context.rstrip(", ;-\n")

                gold_locations = gold_answer.split("/")

                for k in [6, 10, 13, 17, 21]: #range(1, num_templates+1):

                    y = char_count
                    x = y - 2600

                    if x < len_title:
                        text = story[len_title:y]

                    else:
                        x = story[x:y].find(" ") + x
                        text = story[x:y]

                    text = text_clean_ending(text)
                    text = remove_new_lines(text)

                    for num_chap2_tokens in tokens_to_append:
                        chapter2_part = tokenizer.decode(chapter2_tokenized[:num_chap2_tokens], skip_special_tokens=True)
                        chapter2_part = chapter2_part.rstrip(", ;-\n")
                        context = text + "\n" + chapter2_part
                        
                        ## TODO: Answer Generation
                        
                        prompt, context2 = create_prompt_clipped(tokenizer, k, context, character, grammatical_number, 512)
                
                        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

                        with torch.no_grad():
                            output = model.generate(input_ids)

                        out = tokenizer.decode(output[0], skip_special_tokens=True)
                        
                        ## Ends here.
                        match1, match2 = exactly_or_fuzzily_matched(out, gold_locations)
                        result_object = TextInjectionResults(prompt, out, ind, character, gold_locations, k, num_chap2_tokens, story_no, match1, match2)
                        all_result_objects[k].append(result_object)
                        
                i += 1
                print("line:", i)
            else:

                j += 1
                paragraph = paragraphs[j]
                paragraph = paragraph.replace("\n", " ")
                char_count += (len(paragraph) + 2)

    return all_result_objects

path = "litbank/original/105_persuasion.txt"

with open(path, "r") as f:
    book = f.read()
    
indices = []
for i in range(1, 100):
    chapter_header = "Chapter " + str(i) + "\n\n"
    indices.append(book.find(chapter_header))

ind1 = indices[0]
ind2 = indices[1]
i = 2
chapters = []
while ind2 != -1:
    chapters.append(book[ind1:ind2])
    ind1 = ind2
    ind2 = indices[i]
    i += 1 

chapter1 = remove_new_lines(chapters[0])
paragraph_start_ind = chapters[1].find("Mr Shepherd")
chapter2 = remove_new_lines(chapters[1][paragraph_start_ind:])

text_injection_instance = TextInjection(model, tokenizer, device, chapter2, tokens_to_append)

all_result_objects = predictor(text_injection_instance)

with open("Text_injection_flant5xxl_predictions.txt", "wb") as f:
    pickle.dump(all_result_objects, f)