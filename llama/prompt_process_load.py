import os
import json
import random

import torch
import torch.nn.functional as F
#from torch.utils.data import Dataloader

from llama import Llama

dataset_path = '/home1/bidnur/llama/alpaca_dataset/alpaca_data.json'
tokenizer_path = '/project/saifhash_1190/llama2-7b/tokenizer.model'
ckpt_dir = '/project/saifhash_1190/llama2-7b/'

random.seed(0)

def load_prompt_dataset(path, num_samples=200):
    """read json prompts file and return list of num_samples items"""
    f = open(path, "r")
    dataset = json.load(f)
    #print(d[:4])
    f.close()

    if (num_samples < len(dataset)):
        dataset = dataset[:num_samples]

    return dataset

def no_input_prompt(prompt):
    """Extracts prompts with no inputs """
    prompt_str = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                  "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(prompt)
    return prompt_str

def with_input_prompt(prompt):
    """Extracts prompts with inputs """
    prompt_str = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
                  "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format_map(prompt)
    return prompt_str

def format_prompt(prompt):
    """Returns formatted prompt based on input prompt passed"""
    if (prompt['input'] == ''):
        return no_input_prompt(prompt)
    else:
        return with_input_prompt(prompt)

num_samples = 10
prompts_dataset = load_prompt_dataset(dataset_path, num_samples=num_samples)
print(prompts_dataset[:3])

formatted_prompts = [format_prompt(prompt) for prompt in prompts_dataset]

#for x in formatted_prompt[:3]:
#    print(x)


#llama build returns llama class object with loaded model and tokenizer
model_class = Llama.build(ckpt_dir, tokenizer_path, 128, 4)

# access the model and the tokenizer
model = model_class.model
tokenizer = model_class.tokenizer

tokenized_prompt = [tokenizer.encode(prompt, bos = True, eos = False) for prompt in formatted_prompts]
print(tokenized_prompt[:3])

#train_percent = 0.8
#test_percent = 1 - train_percent
#train_data, test_data = tokenized_prompt[:(train_percent * num_samples)], tokenized_prompt[(test_percent * numsamples):]
#
#batch_size = 4
#train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
