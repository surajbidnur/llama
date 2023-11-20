import sys
import torch
import json
from torch.utils.data import Dataset, DataLoader
from llama import Llama
import loralib as lora
from torch import nn
import numpy as np

#DATASET_PATH = 'alpaca_dataset/processed_dataset_small_4096.json'
DATASET_PATH = 'alpaca_dataset/alpaca_data.json'
TOKENIZER_PATH = '/project/saifhash_1190/llama2-7b/tokenizer.model'
CKPT_DIR = '/project/saifhash_1190/llama2-7b'
MAX_SEQ_LEN = 512
MAX_BATCH_SIZE = 1
EPOCHS = 2
PROMPTS = 200

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

def tokenize_training_data(training_data, tokenizer):
    tokenized = [tokenizer.encode(s["example"], bos=False, eos=False) for s in training_data]
    return tokenized

def pad_sequence(sequence, padding_token, max_seq_len=512, position='left'):

    curr_len = len(sequence)

    if curr_len >= max_seq_len:
        return sequence[:max_seq_len]

    padding_size = max_seq_len - curr_len + 1 # input size is max_seq_len + 1

    padding_tokens = [padding_token] * padding_size

    if position == 'left':
        padded_sequence = padding_tokens + sequence # left padding
    elif position == 'right':
        padded_sequence = sequence + padding_tokens # right padding

    return padded_sequence

class PromptDataset(Dataset):
    """Alpaca prompts dataset"""

    #def __init__(self, json_path):
    def __init__(self, prompt_ds):
        self.prompt_ds = prompt_ds
        #with open(json_path) as f:
        #    self.prompt_ds = json.load(f)

    def __len__(self):
        return len(self.prompt_ds)

    def __getitem__(self, idx):
        #input_ids = self.prompt_ds[idx]['input_ids']
        #labels = self.prompt_ds[idx]['labels']
        input_ids = self.prompt_ds[idx][:-1]
        labels = self.prompt_ds[idx][1:]
        sample = {'inputs': torch.tensor(input_ids), 'target': torch.tensor(labels)}

        return sample

llama_class = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE
        )

model = llama_class.model
Tokenizer = llama_class.tokenizer

prompts = load_dataset(DATASET_PATH, n=PROMPTS)
print("-------------------PROMPTS--------------------")
print(prompts[:3])

outputs = [load_outputs(prompt) for prompt in prompts]
print("-------------------OUTPUTS--------------------")
print(outputs[:3])

formatted_prompts = [formatted_prompt(prompt) for prompt in prompts]
print("-------------------FORMATTED_PROMPTS--------------------")
print(formatted_prompts[:3])

dataset = [{"prompt":s, "output":t, "example": s+t} for s,t in zip(formatted_prompts, outputs)]
print("-------------------DATASET--------------------")
print(dataset[:2])

tokenized_data = tokenize_training_data(dataset, Tokenizer)
training_data = [pad_sequence(s, Tokenizer.pad_id, MAX_SEQ_LEN, 'left') for s in tokenized_data]
print("-------------------TRAINING_DATA--------------------")
print(training_data[0][:20], training_data[0][-20:], len(training_data[0]))

#sys.exit()

#train_dataset = PromptDataset(DATASET_PATH)
train_dataset = PromptDataset(training_data)
train_dataloader = DataLoader(train_dataset, batch_size=MAX_BATCH_SIZE, shuffle=False)
d = next(iter(train_dataset))
#print("First iteration dataset\n", d)
print("-------------------TRAINING_DATALOADER--------------------")
print(len(train_dataloader))
DS_LEN = len(train_dataloader)

for i in train_dataloader:
    print("Input shape: {0}, Target shape: {1}".format(i['inputs'].shape, i['target'].shape))
    break

print("-------------------TOTAL_MODEL_PARAMS--------------------")
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total model params: {0}".format(pytorch_total_params))

print("-------------------TOTAL_TRAINABLE_MODEL_PARAMS--------------------")
pytorch_total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable model params: {0}".format(pytorch_total_params_grad))

print("-------------------TRAINABLE_LORA_PARAMS--------------------")
lora.mark_only_lora_as_trainable(model)
pytorch_total_params_grad_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable LoRA params: {0}".format(pytorch_total_params_grad_lora))

#for name, layer in model.named_modules():
#    print(name, layer)
print(model)

print("-------------------TRAINING_STEPS--------------------")
train_steps = EPOCHS * DS_LEN
print("Training Steps: {}".format(train_steps))

#sys.exit()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
#scaler = torch.cuda.amp.GradScaler()
ACC_STEP = 8

def train(epochs=EPOCHS, batch_size=MAX_BATCH_SIZE):
    loss_per_epoch = list()
    for epoch in range(epochs):
        losses = list()
        for i, data in enumerate(train_dataloader):
            inputs, labels = data['inputs'], data['target']

            #with torch.autocast(device_type='cuda', dtype=torch.float16):
            #    #outputs = model.forward(inputs, 0)
            #    outputs = model(inputs, 0)
            outputs = model(inputs, 0)

            #    loss = loss_fn(outputs.transpose(1,2), labels)
            loss = loss_fn(outputs.transpose(1,2), labels)
            #    #_, loss = model.generate(inputs, MAX_SEQ_LEN, logprobs=True)
            #    loss.backward()
            loss.backward()
            #scaler.scale(loss).backward()

            #if i % ACC_STEP == 0:

            #    optimizer.step()
            optimizer.step()
            ##scaler.step(optimizer)

            #    optimizer.zero_grad()
            optimizer.zero_grad()
            #scaler.update()

            losses.append(loss.item())

            print("Loss per batch: {}".format(loss.item()))

        epoch_loss = np.average(losses)
        loss_per_epoch.append(epoch_loss)
        print("Epoch: {0}, Epoch Loss: {1}".format(epoch+1, epoch_loss))

    return loss_per_epoch

print("-------------------BEGINNING_TRAINING--------------------")
loss_list = train()
print("-------------------FINISHED_TRAINING--------------------")
 
