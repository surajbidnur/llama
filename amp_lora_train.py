import sys
import time
import copy
import torch
import json
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from llama import Llama
#from llama.tokenizer import Tokenizer
#from llama.model import ModelArgs, Transformer
from dataclasses import dataclass
import loralib as lora
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = 'alpaca_dataset/alpaca_data.json'
TOKENIZER_PATH = '/project/saifhash_1190/llama2-7b/tokenizer.model'
PARAMS_PATH = '/project/saifhash_1190/llama2-7b/params.json'
CKPT_DIR = '/project/saifhash_1190/llama2-7b'
MODEL_PATH = '/project/saifhash_1190/llama2-7b/consolidated.00.pth'
MAX_SEQ_LEN = 256
MAX_BATCH_SIZE = 8
EPOCHS = 5
PROMPTS = 200
IGNORE_INDEX = -100
LEARNING_RATE = 0.00002

def load_dataset(path, n=200):
    """loads json dataset from filepath. If n=-1, full dataset is read, else only n items are read"""
    f = open(path, "r")
    ds = json.load(f)
    if (n == -1):
        return ds
    else:
        return ds[:n]

def load_outputs(prompt):
    return prompt['output']

def prompt_no_input(prompt):
    txt = ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n"
          "### Response:\n").format_map(prompt)

    return txt

def prompt_with_input(prompt):
    txt = ("Below is an instruction that describes a task, paired with an input that further provides context."
          "Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n"
          "### Input:\n{input}\n\n"
          "### Response:\n").format_map(prompt)

    return txt

def formatted_prompt(prompt):
    if (prompt['input'] == ''):
        return prompt_no_input(prompt)
    else:
        return prompt_with_input(prompt)

def tokenized_dict(tokens):
    input_ids = labels = [tokenized for tokenized in tokens]
    input_ids_len = labels_len = [tokenized.ne(0).sum().item() for tokenized in tokens]

    return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_len=input_ids_len,
            labels_len=labels_len,
            )

def tokenize_data(examples, tokenizer):
    tokenized = [torch.tensor(tokenizer.encode(s, bos=True, eos=True)) for s in examples]
    input_ids = labels = [tokens for tokens in tokenized]
    input_ids_len = labels_len = [tokens.ne(tokenizer.pad_id).sum().item() for tokens in tokenized]

    return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_len=input_ids_len,
            labels_len=labels_len,
            )

def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    tokenized_examples = tokenize_data(examples, tokenizer)
    tokenized_sources = tokenize_data(sources, tokenizer)
    input_ids = tokenized_examples["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, tokenized_sources["input_ids_len"]):
        label[:source_len - 1] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)
    
class SupervisedDataset(Dataset):
    """Alpaca prompts dataset"""

    def __init__(self, json_path, tokenizer, n=200):
        super(SupervisedDataset, self).__init__()

        with open(json_path, "r") as f:
            prompt_ds = json.load(f)

        if n < len(prompt_ds):
            prompt_ds = prompt_ds[:n]

        sources = [formatted_prompt(example) for example in prompt_ds]

        targets = [example['output'] for example in prompt_ds]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    #tokenizer: transformers.PreTrainedTokenizer
    batch_size: int = 1 #default batch size

    def __call__(self, instances):
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                        input_ids, batch_first=True, padding_value=-1
                        )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(-1),
                )

def make_supervised_data_module(tokenizer, data_path, prompts):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_path, tokenizer, n=prompts)
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train(model, dataloader):

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2000, eta_min=0.1*LEARNING_RATE) 

    model.train()

    loss_list = list()
    gpu_mem_list = list()
    perplexity_list = list()

    for epoch in range(EPOCHS):

        start_time = time.time()

        train_loss = 0
        perplexity = 0

        with tqdm(dataloader, desc=f"Epoch: {epoch + 1}/{EPOCHS}", ascii=' >=') as pbar:
            for i, data in enumerate(pbar):
                inputs, labels, mask = data['input_ids'].to(device), data['labels'].to(device), data['attention_mask'].to(device)

                optimizer.zero_grad()

                with torch.autocast(device_type="cuda"):
                    outputs = model(inputs, 0)

                    shift_logits = outputs[...,:-1,:].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_logits = shift_logits.view(-1, 32000)
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)

                    loss = loss_fn(shift_logits, shift_labels)
                    assert loss.dtype is torch.float32
                
                loss.backward()

                clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step(epoch + i / iters)

                pbar.set_postfix(train_loss=F"{loss.item():.6f}", perplexity=f"{torch.exp(loss):.6f}")
                pbar.update()

                train_loss += loss.item()

        gpu_mem_list.append(torch.cuda.memory_allocated(device='cuda') / (1024 ** 3))
        print("Max GPU memory usage: {0:.2f}GB".format(max(gpu_mem_list)))

        avg_loss = train_loss / len(dataloader)
        loss_list.append(avg_loss)
        perplexity_list.append(torch.exp(torch.tensor(avg_loss)).tolist())
        print("Average training loss: {0:.6f}, Perplexity: {1:.6f}".format(avg_loss, torch.exp(torch.tensor(avg_loss)).tolist()))

        print("Epoch time {0:.2f} seconds".format(time.time() - start_time))

    return loss_list, gpu_mem_list, perplexity_list

def evaluate(llama_class, prompts, seq_len=128):
    llama_class.model.eval()
    results = llama_class.text_completion(prompts, max_gen_len=seq_len)

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n=======================================\n")

def plot_graph(x, y, name):
    plt.plot(x, y, 'o-r')
    plt.xlabel('epochs')

    if 'loss' in name:
        plt.title("Training Loss vs Epochs")
        plt.ylabel(name + '(lora_amp)')
    if 'mem' in name:
        plt.title("GPU memory usage (GB) vs Epochs")
        plt.ylabel(name + '(lora_amp)')
    if 'perplexity' in name:
        plt.title("Perplexity vs Epochs")
        plt.ylabel(name + '(lora_amp)')

    plt.savefig(name + '.png')

if __name__ == "__main__":

    print("---------------EXECUTING AMP LORA LLAMA TRAINING----------------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    llama_class = Llama.build(
            ckpt_dir=CKPT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE
            )

    model = llama_class.model
    model.to(device)

    tokenizer = llama_class.tokenizer

    print(model)

    print("-------------------TOTAL_MODEL_PARAMS--------------------")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total model params: {0}".format(pytorch_total_params))

    print("-------------------TOTAL_TRAINABLE_MODEL_PARAMS--------------------")
    pytorch_total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable model params: {0}".format(pytorch_total_params_grad))

    lora.mark_only_lora_as_trainable(model)
    print("-------------------TRAINABLE_LORA_PARAMS--------------------")
    pytorch_total_params_grad_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable LoRA params: {0}".format(pytorch_total_params_grad_lora))

    data_module = make_supervised_data_module(tokenizer, DATASET_PATH, PROMPTS)
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator)
    iters = len(train_dataloader)

    loss, mem, ppl = train(model, train_dataloader)
    print("Average loss: {0:.4f}, Avg GPU mem usage: {1:.2f}GB".format(sum(loss) / len(loss), sum(mem) / len(mem)))

    plot_graph([x+1 for x in range(EPOCHS)], loss, 'loss')
    plot_graph([x+1 for x in range(EPOCHS)], mem, 'gpu_mem')
    plot_graph([x+1 for x in range(EPOCHS)], ppl, 'perplexity')

    prompts = ["Generate a list of 10 items a person might need for a camping trip",
            "Who is the world's most famous painter?",
            "Describe the function of a computer motherboard",
            "What does DNA stand for",
            "Charles Darwin's natural selection states that",
            "The Hubble Space Telescope is a",
            "The legend of Ramayana is",
            """An inaugral address to the 2023 Diwali celebrations:
            
            Deviyo aur sajjano,
            
            Welcome to the """,]

    evaluate(llama_class, prompts, 64)

