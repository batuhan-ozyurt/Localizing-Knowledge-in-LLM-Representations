from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

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

import matplotlib.pyplot as plt
from math import log

def predictor(text_injection_instance, write_in_file=False, out_path_prefix="Text_injection_GPT2-XL_"):
    
    # prompt has the max number of tokens: 1024, and we start at a " " char.
    
    model = text_injection_instance.model
    tokenizer = text_injection_instance.tokenizer
    device = text_injection_instance.device
    chapter1 = text_injection_instance.chapter1
    chapter2_tokenized = text_injection_instance.chapter2_tokenized
    tokens_to_append = text_injection_instance.tokens_to_append

    predictions = {}
    all_result_objects = defaultdict(list)

    for item in dir_list_annotations:
        
        print(item)
        
        story_no = item[len("Andersen_story"):-len(".txt")]

        f = open(os.path.join(path_andersen, item), 'r') 
        story = f.read()
        f.close()
        
        predictions[item] = defaultdict(dict)

        if write_in_file:
            out_path = out_path_prefix + item[:-3] + "xlsx"
            writer = pd.ExcelWriter(out_path, engine='xlsxwriter')
            workbook = writer.book
            format = workbook.add_format({'text_wrap': True})

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

                character = line[1]
                gold_answer = line[2]
                grammatical_number = line[3]
                context = paragraph
                context = context.rstrip(", ;-\n")

                gold_locations = gold_answer.split("/")
                my_dic = {"Prompts": [gold_answer, "-", "-", "-"]}

                for k in range(1, num_llama_templates+1):#range(1, num_templates+1):

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
                        prompt = completion_prompt_generator(tokenizer, k, context, character, grammatical_number, 1024)
                        """inputs = tokenizer.encode(prompt, return_tensors="pt")
                        inputs = inputs.to(device)
                        with torch.no_grad():
                            outputs = model.generate(inputs)
                        out = tokenizer.decode(outputs[0], skip_special_tokens=True)"""
                        out = model(prompt, max_new_tokens=40, num_return_sequences=1)[0]["generated_text"]
                        if out[:len(prompt)] == prompt:
                            out = out[len(prompt)+1:]
                        match1, match2 = exactly_or_fuzzily_matched(out, gold_locations)
                        predictions[item][k][num_chap2_tokens] = out
                        result_object = TextInjectionResults(prompt, out, gold_locations, character, k, num_chap2_tokens, story_no, match1, match2)
                        all_result_objects[k].append(result_object)
                        if write_in_file:
                            my_dic[prompt] = [out, num_chap2_tokens, match1, match2]
                        break
                    break
                break

                if write_in_file:
                    df = pd.DataFrame(data=my_dic, index=["output", "num_tokens_appended", "exact match?", "fuzzy match?"])
                    df = (df.T)
                    df.to_excel(writer, sheet_name=str(i+1))
                    worksheet = writer.sheets[str(i+1)]
                    for idx, col in enumerate(df):
                        max_len = 75
                        worksheet.set_column(idx, idx, max_len, format)

                i += 1
                print("line:", i)
            else:

                j += 1
                paragraph = paragraphs[j]
                paragraph = paragraph.replace("\n", " ")
                char_count += (len(paragraph) + 2)

        if write_in_file:
            writer.save()

    return predictions, all_result_objects

all_annotations = {}

for item in dir_list_annotations:
    
    f = open(os.path.join(path_annotations, item), 'r')
    annotations = pd.read_csv(f, sep="\t")
    annotations = annotations.values
    f.close()
    
    all_annotations[item] = annotations
    
class TextInjection():
    def __init__(self, model, tokenizer, device, chapter1, chapter2, tokens_to_append):
        self.model = model
        self.tokenizer = tokenizer
        #self.model = self.model.to(device)
        self.device = device
        self.chapter1 = chapter1
        self.chapter2_tokenized = tokenizer.encode(chapter2)
        self.tokens_to_append = tokens_to_append
        
class TextInjectionResults():
    def __init__(self, prompt, model_out, gold_locations, character, k, tokens_appended, story_no, exact_match, fuzzy_match):
        self.prompt = prompt
        self.model_out = model_out
        self.gold_locations = gold_locations
        self.character = character
        self.k = k
        self.tokens_appended = tokens_appended
        self.story_no = story_no
        self.exact_match = exact_match
        self.fuzzy_match = fuzzy_match

path = "/scratch/users/bozyurt20/hpc_run/litbank/original/105_persuasion.txt"

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
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
print("tokenizer")
model = pipeline('text-generation', model='gpt2-xl', device=device)
print("model")

tokens_to_append = [x for x in range(0, 1001, 40)]

chapter1 = remove_new_lines(chapters[0])
paragraph_start_ind = chapters[1].find("Mr Shepherd")
chapter2 = remove_new_lines(chapters[1][paragraph_start_ind:])

text_injection_instance = TextInjection(model, tokenizer, device, chapter1, chapter2, tokens_to_append)

predictions, all_result_objects = predictor(text_injection_instance, write_in_file=False, out_path_prefix="Text_injection_GPT2-XL_")

with open("Text_injection_GPT2-XL_predictions.txt", "wb") as f:
    pickle.dump(predictions, f)
    
with open("Text_injection_GPT2-XL_all_result_objects.txt", "wb") as f:
    pickle.dump(all_result_objects, f)
