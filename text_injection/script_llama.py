from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

from typing import Tuple
import sys
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def predictor(text_injection_instance, write_in_file=False, out_path_prefix="Text_injection_llama_13B_", temperature=0.8, top_p=0.95):
    
    model = text_injection_instance.model
    tokenizer = text_injection_instance.tokenizer
    device = text_injection_instance.device
    chapter1 = text_injection_instance.chapter1
    chapter2_tokenized = text_injection_instance.chapter2_tokenized
    tokens_to_append = text_injection_instance.tokens_to_append

    predictions = {}
    all_result_objects = {}
    for k in range(1, num_llama_templates+1):
        all_result_objects[k] = []

    for item in dir_list_annotations:
        
        print(item)
        
        story_no = item[len("Andersen_story"):-len(".txt")]

        f = open(os.path.join(path_andersen, item), 'r') 
        story = f.read()
        f.close()
        
        predictions[item] = {}
        for k in range(1, num_llama_templates+1):
            predictions[item][k] = {}

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

                for k in [1]: #range(1, num_templates+1):

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
                        chapter2_part = tokenizer.decode(chapter2_tokenized[:num_chap2_tokens])
                        context = text + "\n" + chapter2_part
                        context = text_clean_ending(context)
                        #start = time.time()
                        prompt = llama_prompt_generator(tokenizer, k, context, character, grammatical_number, 2048)
                        #print("Time to generate prompt:", time.time() - start )
                        #start = time.time()
                        results = model.generate(
                            [prompt], max_gen_len=512, temperature=temperature, top_p=top_p
                        )
                        #print("Time to generate output:", time.time() - start )
                        # inputs = tokenizer.encode(prompt, return_tensors="pt")
                        # inputs = inputs.to(device)
                        # with torch.no_grad():
                        #     outputs = model.generate(inputs)
                        # out = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        out = results[0]
                        match1, match2 = exactly_or_fuzzily_matched(out, gold_locations)
                        predictions[item][k][num_chap2_tokens] = out
                        result_object = TextInjectionResults(prompt, out, gold_locations, character, k, num_chap2_tokens, story_no, match1, match2)
                        all_result_objects[k].append(result_object)
                        if write_in_file:
                            my_dic[prompt] = [out, num_chap2_tokens, match1, match2]
                 

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
        self.chapter2_tokenized = tokenizer.encode(chapter2, bos=True, eos=False)
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator, tokenizer


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 2048,
    max_batch_size: int = 1,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator, tokenizer = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    path = "/kuacc/users/bozyurt20/hpc_run/105_persuasion.txt"

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
        
    #tokens_to_append = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    tokens_to_append = [x for x in range(0, 2001, 100)]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    chapter1 = remove_new_lines(chapters[0])
    paragraph_start_ind = chapters[1].find("Mr Shepherd")
    chapter2 = remove_new_lines(chapters[1][paragraph_start_ind:])

    text_injection_instance = TextInjection(generator, tokenizer, device, chapter1, chapter2, tokens_to_append)

    #prompts = [list of strings]

    predictions, all_result_objects = predictor(text_injection_instance, write_in_file=False, out_path_prefix="Text_injection_llama_13B")

    with open("Text_injection_llama_13B_predictions.txt", "wb") as f:
        pickle.dump(predictions, f)
        
    with open("Text_injection_llama_13B_all_result_objects.txt", "wb") as f:
        pickle.dump(all_result_objects, f)

if __name__ == "__main__":
    fire.Fire(main)
