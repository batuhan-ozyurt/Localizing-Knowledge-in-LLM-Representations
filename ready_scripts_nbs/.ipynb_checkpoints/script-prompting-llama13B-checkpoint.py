#torchrun --nproc_per_node 2 /kuacc/users/bozyurt20/hpc_run/ready_scripts_nbs/script-prompting-llama13B.py --ckpt_dir /userfiles/bozyurt20/llama/llama_weights/13B --tokenizer_path /userfiles/bozyurt20/llama/llama_weights/tokenizer.model

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

from typing import Tuple
import fire
import time
import json
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA

import sys
sys.path.append('/scratch/users/bozyurt20/hpc_run/utilities')
from util_research import *

num_templates = 3

path_andersen = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Andersen"
path_fanny = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Fanny Fern"
path_annotations = "/kuacc/users/bozyurt20/hpc_run/ChildrenStories/Annotations"

dir_list_andersen = os.listdir(path_andersen)
dir_list_fanny = os.listdir(path_fanny)
dir_list_annotations = os.listdir(path_annotations)

stop_words = set(stopwords.words("english"))

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


def predictor(model_instance, temperature=0.8, top_p=0.95):
    
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
                
                start_time = time.time()
                
                prompt = llama_prompt_generator(tokenizer, k, text, character, grammatical_number, 2048)
                
                with torch.no_grad():
                    results = model.generate(
                                [prompt], max_gen_len=512, temperature=temperature, top_p=top_p
                            )
                out = results[0]
                
                print("One generation:", time.time() - start_time)
                
                match1, match2 = exactly_or_fuzzily_matched(out, character, gold_locations)
                result_object = ResultObject(prompt, out, ind, character, gold_locations, story_no, k, match1, match2)
                all_result_objects[k].append(result_object)
                
    return all_result_objects

    
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
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_instance = ModelInstance(generator, tokenizer, device)

    all_result_objects = predictor(model_instance, temperature=temperature, top_p=top_p)
        
    with open("llama_13B_predictions.txt", "wb") as f:
        pickle.dump(all_result_objects, f)

if __name__ == "__main__":
    fire.Fire(main)
    
