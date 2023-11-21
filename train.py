import sys
import copy
import torch
import json
from torch.utils.data import Dataset, DataLoader
from llama import Llama
from dataclasses import dataclass
#import loralib as lora
from torch import nn
import numpy as np

#DATASET_PATH = 'alpaca_dataset/processed_dataset_small_4096.json'
DATASET_PATH = 'alpaca_dataset/alpaca_data.json'
TOKENIZER_PATH = '/project/saifhash_1190/llama2-7b/tokenizer.model'
CKPT_DIR = '/project/saifhash_1190/llama2-7b'
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 1
EPOCHS = 2
PROMPTS = 20
IGNORE_INDEX = -1

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


def pad_sequence(sequence, padding_token, max_seq_len=512, position='left'):

    curr_len = len(sequence)

    if curr_len >= max_seq_len:
        return sequence[:max_seq_len]

    padding_size = max_seq_len - curr_len + 1 # input size is max_seq_len + 1

    padding_tokens = [padding_token] * padding_size

    if position == 'left':
        padded_sequence = padding_tokens
        padded_sequence.extend(sequence) # left padding
    elif position == 'right':
        padded_sequence = sequence.copy()
        padded_sequence.extend(padding_tokens) # right padding

    return padded_sequence

def tokenized_dict(tokens):
    input_ids = labels = [tokenized for tokenized in tokens]
    input_ids_len = labels_len = [tokenized.ne(0).sum().item() for tokenized in tokens]

    return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_len=input_ids_len,
            labels_len=labels_len,
            )

def tokenize_data(examples, tokenizer, pad_token=0, max_len=512, pad_position='right'):
    #tokenized = [torch.tensor(tokenizer.encode(pad_sequence(s, pad_token, max_len, pad_position), bos=False, eos=False)) for s in examples]
    tokenized = [tokenizer.encode(s, bos=False, eos=False) for s in examples]
    pad_tokenized = [torch.tensor(pad_sequence(s, pad_token, max_len, pad_position)) for s in tokenized]
    #print(tokenized[:2])
    #sys.exit()
    input_ids = labels = [tokens for tokens in pad_tokenized]
    input_ids_len = labels_len = [tokens.ne(0).sum().item() for tokens in pad_tokenized]

    return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_len=input_ids_len,
            labels_len=labels_len,
            )

def preprocess(sources, targets, tokenizer):
    examples = [s+t for s,t in zip(sources, targets)]
    tokenized_examples = tokenize_data(examples, tokenizer)
    tokenized_sources = tokenize_data(sources, tokenizer)
    input_ids = tokenized_examples["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, tokenized_sources["input_ids_len"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)
    
class SupervisedDataset(Dataset):
    """Alpaca prompts dataset"""

    def __init__(self, json_path, tokenizer, n=200):
        super(SupervisedDataset, self).__init__()
    #def __init__(self, prompt_ds):
        #self.prompt_ds = prompt_ds
        with open(json_path, "r") as f:
            prompt_ds = json.load(f)

        if n < len(prompt_ds):
            prompt_ds = prompt_ds[:n]

        #prompt_input, prompt_no_input = prompt_dict["prompt_input"], prompt_dict["prompt_no_input"]

        #sources = [
        #        prompt_no_input.format_map(example) if example["input"] == "" else prompt_input.format_map(example)
        #        for example in prompt_ds
        #        ]
        sources = [prompt_no_input(example) if example["input"] == "" else prompt_with_input(example) for example in prompt_ds]

        targets = [f"{example['output']}{tokenizer.eos_id}" for example in prompt_ds]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        #input_ids = self.prompt_ds[idx]['input_ids']
        #labels = self.prompt_ds[idx]['labels']
        #input_ids = self.prompt_ds[idx][:-1]
        #labels = self.prompt_ds[idx][1:]
        #sample = {'inputs': input_ids, 'target': labels}
        #sample = {'inputs': torch.tensor(input_ids), 'target': torch.tensor(labels)}
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

        #return sample

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    #tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                        input_ids, batch_first=True, padding_value=0
                        )
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=input_ids.ne(0),
                )

def make_supervised_data_module(tokenizer, data_path):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_path, tokenizer, n=PROMPTS)
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train(ds_path, tokenizer_path, ckpt_path):

    llama_class = Llama.build(
            ckpt_dir=ckpt_path,
            tokenizer_path=tokenizer_path,
            max_seq_len=256,
            max_batch_size=1
            )

    model = llama_class.model
    Tokenizer = llama_class.tokenizer

    #freeze most of the model param to save memory for debugging purpose
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False

    
    data_module = make_supervised_data_module(Tokenizer, ds_path)
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]

    train_dataloader = DataLoader(train_dataset, batch_size=MAX_BATCH_SIZE, shuffle=False, collate_fn=data_collator)

    print(train_dataset[0])
    print(train_dataset[0]['input_ids'].shape, train_dataset[0]['labels'].shape)
    #print(Tokenizer.decode(train_dataset[0]['input_ids']))

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    loss_per_epoch = list()
    for epoch in range(EPOCHS):
        losses = list()
        #torch.set_grad_enabled(True)
        for i, data in enumerate(train_dataloader):
            #inputs_tensor = torch.tensor(data['inputs'], dtype=torch.long)
            #labels_tensor = torch.tensor(data['target'], dtype=torch.long)
            #print(inputs_tensor.shape)
            inputs, labels = data['input_ids'], data['labels']
            inputs, labels = inputs[:, :-1], labels[:, 1:]
            print(inputs.shape, labels.shape)

            #with torch.autocast(device_type='cuda', dtype=torch.float16):
                #outputs = model.forward(inputs, 0)
                #outputs = model(inputs, 0)
            #outputs = model(inputs_tensor, 0)
            outputs = model.forward(inputs, 0)
            print(outputs.shape)

                #loss = loss_fn(outputs.transpose(1,2), labels)
            #loss = loss_fn(outputs.transpose(1,2), labels_tensor)
            loss = loss_fn(outputs.transpose(1,2), labels)
            #print(labels)
                #_, loss = model.generate(inputs, MAX_SEQ_LEN, logprobs=True)
                #loss.backward()
            loss.backward()
            #scaler.scale(loss).backward()

            #if i % ACC_STEP == 0:

            #    optimizer.step()
            optimizer.step()
            #scaler.step(optimizer)

            #    optimizer.zero_grad()
            optimizer.zero_grad()
            #scaler.update()

            losses.append(loss.item())
            print("Loss per batch: {}".format(loss.item()))

        epoch_loss = np.average(losses)
        loss_per_epoch.append(epoch_loss)
        print("Epoch: {0}, Epoch Loss: {1}".format(epoch+1, epoch_loss))


if __name__ == "__main__":
    train(DATASET_PATH, TOKENIZER_PATH, CKPT_DIR)

#prompts = load_dataset(DATASET_PATH, n=PROMPTS)
#print("-------------------PROMPTS--------------------")
#print(prompts[:3])
#
#outputs = [load_outputs(prompt) for prompt in prompts]
#print("-------------------OUTPUTS--------------------")
#print(outputs[:3])
#
#sources = [formatted_prompt(prompt) for prompt in prompts]
#print("-------------------FORMATTED_PROMPTS--------------------")
#print(formatted_prompts[:3])
#
#print("-------------------DATASET--------------------")
#print(dataset[:2])
#
#
##training_data = [pad_sequence(s, 0, MAX_SEQ_LEN, 'right') for s in tokenized_data]
#print("-------------------TRAINING_DATA--------------------")
#print(training_data[0][:20], training_data[0][-20:], len(training_data[0]))
#
##sys.exit()
#
##train_dataset = PromptDataset(DATASET_PATH)
#train_dataset = PromptDataset(training_data)
#train_dataloader = DataLoader(train_dataset, batch_size=MAX_BATCH_SIZE, shuffle=False, collate_fn=torch.utils.data.default_collate)
##d = next(iter(train_dataset))
##print("First iteration dataset\n", d)
#print("-------------------TRAINING_DATALOADER--------------------")
#print(len(train_dataloader))
#DS_LEN = len(train_dataloader)
#
#for i in train_dataloader:
#    print("Input shape: {0}, Target shape: {1}".format(i['inputs'].shape, i['target'].shape))
#    break
#
#print("-------------------TOTAL_MODEL_PARAMS--------------------")
#pytorch_total_params = sum(p.numel() for p in model.parameters())
#print("Total model params: {0}".format(pytorch_total_params))
#
#print("-------------------TOTAL_TRAINABLE_MODEL_PARAMS--------------------")
#pytorch_total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print("Trainable model params: {0}".format(pytorch_total_params_grad))
#
#print("-------------------TRAINABLE_LORA_PARAMS--------------------")
##for param in model.parameters():
##    param.require_grad = False
##x = sum(p.numel() for p in model.parameters() if p.requires_grad)
##print("Trainable params after freezing: {}".format(x))
##assert x == 0, "Error, some parameters still trainable"
##lora.mark_only_lora_as_trainable(model)
##pytorch_total_params_grad_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
##print("Trainable LoRA params: {0}".format(pytorch_total_params_grad_lora))
#
##for name, layer in model.named_modules():
##    print(name, layer)
#print(model)
#
#print("-------------------TRAINING_STEPS--------------------")
#train_steps = EPOCHS * DS_LEN
#print("Training Steps: {}".format(train_steps))
#
##sys.exit()
#
#optimizer = torch.optim.Adam(model.parameters())
#loss_fn = nn.CrossEntropyLoss()
##scaler = torch.cuda.amp.GradScaler()
#ACC_STEP = 8
#
#def train(epochs=EPOCHS, batch_size=MAX_BATCH_SIZE):
#    loss_per_epoch = list()
#    for epoch in range(epochs):
#        losses = list()
#        #torch.set_grad_enabled(True)
#        for i, data in enumerate(train_dataloader):
#            #inputs_tensor = torch.tensor(data['inputs'], dtype=torch.long)
#            #labels_tensor = torch.tensor(data['target'], dtype=torch.long)
#            #print(inputs_tensor.shape)
#            inputs, labels = data['inputs'], data['target']
#            print(inputs.shape, labels.shape)
#
#            #with torch.autocast(device_type='cuda', dtype=torch.float16):
#                #outputs = model.forward(inputs, 0)
#                #outputs = model(inputs, 0)
#            #outputs = model(inputs_tensor, 0)
#            outputs = model(inputs, 0)
#            print(outputs.shape)
#
#                #loss = loss_fn(outputs.transpose(1,2), labels)
#            #loss = loss_fn(outputs.transpose(1,2), labels_tensor)
#            loss = loss_fn(outputs.transpose(1,2), labels)
#            #print(labels)
#                #_, loss = model.generate(inputs, MAX_SEQ_LEN, logprobs=True)
#                #loss.backward()
#            loss.backward()
#            #scaler.scale(loss).backward()
#
#            #if i % ACC_STEP == 0:
#
#            #    optimizer.step()
#            optimizer.step()
#            #scaler.step(optimizer)
#
#            #    optimizer.zero_grad()
#            optimizer.zero_grad()
#            #scaler.update()
#
#            losses.append(loss.item())
#
#            print("Loss per batch: {}".format(loss.item()))
#
#        epoch_loss = np.average(losses)
#        loss_per_epoch.append(epoch_loss)
#        print("Epoch: {0}, Epoch Loss: {1}".format(epoch+1, epoch_loss))
#
#    return loss_per_epoch
#
#print("-------------------BEGINNING_TRAINING--------------------")
#loss_list = train()
#print("-------------------FINISHED_TRAINING--------------------")
# 
