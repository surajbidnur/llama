import torch
import json
from torch.utils.data import Dataset, DataLoader
from llama import Llama
import loralib as lora
from torch import nn


DATASET_PATH = 'alpaca_dataset/processed_dataset_small_4096.json'
TOKENIZER_PATH = '/project/saifhash_1190/llama2-7b/tokenizer.model'
CKPT_DIR = '/project/saifhash_1190/llama2-7b'
MAX_SEQ_LEN = 4096
MAX_BATCH_SIZE = 2
EPOCHS = 2

class PromptDataset(Dataset):
    """Alpaca prompts dataset"""

    #def __init__(self, prompt_ds):
    def __init__(self, json_path):
        #self.prompt_ds = prompt_ds
        #self.prompt_ds = json.load(json_path)
        with open(json_path) as f:
            self.prompt_ds = json.load(f)

    def __len__(self):
        return len(self.prompt_ds)

    def __getitem__(self, idx):
        input_ids = self.prompt_ds[idx]['input_ids']
        labels = self.prompt_ds[idx]['labels']
        sample = {'inputs': torch.tensor(input_ids), 'target': torch.tensor(labels)}

        return sample


# load proccessed_dataset from disk
#with open(DATASET_PATH, 'r') as f:
#    prompt_ds = json.load(f)

#print(dataset[:2])
#print(len(dataset))

#for item in dataset:
#    item['input_ids'] = torch.tensor(item['input_ids'])
#    item['labels'] = torch.tensor(item['labels'])

#print(dataset['input_ids'][0].shape)
#train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
#train_dataset = PromptDataset(prompt_ds)
train_dataset = PromptDataset(DATASET_PATH)
train_dataloader = DataLoader(train_dataset, batch_size=MAX_BATCH_SIZE, shuffle=False)
#d = next(iter(train_dataset))
#print("First iteration dataset\n", d)
print(len(train_dataloader))
ds_len = len(train_dataloader)

for i in train_dataloader:
    print("Input shape: {0}, Target shape: {1}".format(i['inputs'].shape, i['target'].shape))

llama_class = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE
        )

model = llama_class.model
tokenizer = llama_class.tokenizer

print("=====================================")
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Total model params: {0}".format(pytorch_total_params))
pytorch_total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable model params: {0}".format(pytorch_total_params_grad))
lora.mark_only_lora_as_trainable(model)
pytorch_total_params_grad_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Trainable LoRA params: {0}".format(pytorch_total_params_grad_lora))

#for name, layer in model.named_modules():
#    print(name, layer)

train_steps = EPOCHS * ds_len

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

def train_one_epoch(epoch_idx):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_dataloader):
        inputs, labels = data['inputs'], data['target']
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        #_, loss = model.generate(inputs, MAX_SEQ_LEN, logprobs=True)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 2 == 0:
            last_loss = running_loss / 2 # loss per batch
            print("  batch {} loss: {}".format(i+1, last_loss))
            running_loss = 0.0

    return last_loss

epoch_number = 0

#model.train()

print("--------------------------------------")
print("BEGINNING TRAINING")

torch.set_grad_enabled(True)

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch + 1))
    #model.train()
    for step, data in enumerate(train_dataloader):
        inputs, labels = data['inputs'], data['target']
        #outputs, _  = llama_class.generate(inputs, MAX_SEQ_LEN, logprobs=False)
        #outputs = model.forward(inputs, 0)
        outputs = model(inputs, 0)
        #outputs = torch.softmax(outputs, dim=-1)
        outputs = torch.argmax(outputs, dim=-1)
        
        #outputs = model.forward(, 0)
        print(outputs.shape, labels.shape)

        #outputs = torch.tensor(outputs)
        loss = -loss_fn(outputs, labels)
        

        #model.zero_grad()
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        #running_loss += loss.item()

        #if i % 2 == 0:
            #last_loss = running_loss / 2 # loss per batch
        print("Training Loss for 1 batch at step : {}".format(loss))

print("TRAINING DONE")
print("--------------------------------------")

    
