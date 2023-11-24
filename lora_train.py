import sys
import copy
import torch
import json
from torch.utils.data import Dataset, DataLoader
from llama import Llama
from dataclasses import dataclass
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import loralib as lora

#DATASET_PATH = 'alpaca_dataset/processed_dataset_small_4096.json'
DATASET_PATH = 'alpaca_dataset/alpaca_data.json'
TOKENIZER_PATH = '/project/saifhash_1190/llama2-7b/tokenizer.model'
CKPT_DIR = '/project/saifhash_1190/llama2-7b'
MAX_SEQ_LEN = 256
MAX_BATCH_SIZE = 1
EPOCHS = 2
PROMPTS = 20
IGNORE_INDEX = -100

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


#def pad_sequence(sequence, padding_token, max_seq_len=MAX_SEQ_LEN, position='left'):
#
#    curr_len = len(sequence)
#
#    if curr_len >= max_seq_len:
#        return sequence[:max_seq_len]
#
#    padding_size = max_seq_len - curr_len + 1 # input size is max_seq_len + 1
#
#    padding_tokens = [padding_token] * padding_size
#
#    if position == 'left':
#        padded_sequence = padding_tokens
#        padded_sequence.extend(sequence) # left padding
#    elif position == 'right':
#        padded_sequence = sequence.copy()
#        padded_sequence.extend(padding_tokens) # right padding
#
#    return padded_sequence

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

        #return sample

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

def make_supervised_data_module(tokenizer, data_path, batch_size):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(data_path, tokenizer, n=PROMPTS)
    data_collator = DataCollatorForSupervisedDataset(batch_size)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

llama_class = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE
        )

model = llama_class.model
model.to(device)
Tokenizer = llama_class.tokenizer
print(model)

print("-------------------TOTAL_MODEL_PARAMS--------------------")
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total model params: {0}".format(pytorch_total_params))

lora.mark_only_lora_as_trainable(model)
print("-------------------TRAINABLE_LORA_PARAMS--------------------")
pytorch_total_params_grad_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable LoRA params: {0}".format(pytorch_total_params_grad_lora))

#freeze most of the model param to save memory for debugging purpose
#for name, param in model.named_parameters():
#    if "lm_head" not in name:
#        param.requires_grad = False


data_module = make_supervised_data_module(Tokenizer, DATASET_PATH, MAX_BATCH_SIZE)
train_dataset = data_module["train_dataset"]
data_collator = data_module["data_collator"]

train_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=data_collator)

#print(train_dataset[0]['input_ids'].shape, train_dataset[0]['labels'].shape)

def train(model, dataloader):

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    loss_per_epoch = list()
    for epoch in range(EPOCHS):
        losses = list()
        #torch.set_grad_enabled(True)
        for i, data in enumerate(dataloader):
            inputs, labels = data['input_ids'].to(device), data['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(inputs, 0)

            shift_logits = outputs[...,:-1,:].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, 32000)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)

            loss = loss_fn(shift_logits, shift_labels)
            #loss.requires_grad=True
            #loss.retain_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            print("Loss per batch: {}".format(loss.item()))

        epoch_loss = np.average(losses)
        loss_per_epoch.append(epoch_loss)
        print("=======================================")
        print("Epoch: {0}, Epoch Loss: {1}".format(epoch+1, epoch_loss))

#def analyze_layer_weights(layer: nn.Module):
#    # Extract and flatten the weights of the given layer
#    #layer_weights = layer.weight.data.cpu().view(-1)
#    layer_weights = layer.data.cpu().view(-1)
#    
#    # Get layer type (Conv2d or Linear) for better title
#    layer_type = type(layer).__name__
#
#    # Plot a histogram of the flattened layer weights
#    plt.hist(layer_weights, density=True, bins=50)
#    plt.title(f"{layer_type} Layer Weights Histogram")
#    plt.xlabel("Weight Value")
#    plt.ylabel("Density")
#    plt.show()
#
#    # Calculate the upper and lower bounds of the range within 3 standard deviations
#    layer_weights_3sigma_max = (layer_weights.mean() + 3 * layer_weights.std()).item()
#    layer_weights_3sigma_min = (layer_weights.mean() - 3 * layer_weights.std()).item()
#
#    # Calculate the range of weights and the 3-sigma range for the layer
#    weight_range = layer_weights.max() - layer_weights.min()
#    sigma_range = layer_weights_3sigma_max - layer_weights_3sigma_min
#
#    print(f"{layer_type} Layer Weight Range: {weight_range.item()}")
#    print(f"{layer_type} Layer 3-Sigma Range: {sigma_range}")

if __name__ == "__main__":
    train(model, train_dataloader)

    #for name, param in model.named_parameters():
    #    #if 'wq' in name:
    #        #analyze_layer_weights(param)
    #        #print(name)
    #        #print(param.shape)
    #    print(name, param.shape)
    #    #print(param.shape)

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
