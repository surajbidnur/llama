import json
import os

from llama import Llama
import loralib as lora

DATASET_PATH = 'alpaca_dataset/alpaca_data.json'
TOKENIZER_PATH = '/project/saifhash_1190/llama2-7b/tokenizer.model'
CKPT_DIR = '/project/saifhash_1190/llama2-7b'
MAX_SEQ_LEN = 2048
MAX_BATCH_SIZE = 4

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

prompts = load_dataset(DATASET_PATH, n=200)
#prompts = load_dataset(DATASET_PATH, n=-1)
print(prompts[:3])
#outputs = [load_outputs(prompt) + '</s>' for prompt in prompts]
outputs = [load_outputs(prompt) for prompt in prompts]
print(outputs[:3])

def formatted_prompt(prompt):
    if (prompt['input'] == ''):
        return prompt_no_input(prompt)
    else:
        return prompt_with_input(prompt)

formatted_prompts = [formatted_prompt(prompt) for prompt in prompts]
print(formatted_prompts[:3])

dataset = [{"prompt":s, "output":t, "example": s+t} for s,t in zip(formatted_prompts, outputs)]
print(dataset[:2])

llama_class = Llama.build(
        ckpt_dir=CKPT_DIR,
        tokenizer_path=TOKENIZER_PATH,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE
        )

model = llama_class.model
tokenizer = llama_class.tokenizer

def pack_dataset(dataset, tokenizer, max_seq_len=1024):
    print(len(dataset))
    tokenized = [tokenizer.encode(s["example"], bos=False, eos=True) for s in dataset]
    print(len(tokenized))

    all_tkn_ids = list()
    for tkn_inp in tokenized:
        all_tkn_ids.extend(tkn_inp)

    print("Total num tokens: {0}".format(len(all_tkn_ids)))
    padded_length = max_seq_len + 1
    packed_ds = list()
    #for i in range(0, len(all_tkn_ids), mx_seq_len + 1):
    for i in range(0, len(all_tkn_ids), padded_length):
        #input_ids = all_tkn_ids[i: i + max_seq_len + 1]
        input_ids = all_tkn_ids[i: i + padded_length]
        #if (len(input_ids) == (max_seq_len + 1):
        if (len(input_ids) < padded_length):
                #packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})
            input_ids += [tokenizer.eos_id] * (padded_length - len(input_ids))

        packed_ds.append({"input_ids": input_ids[:-1], "labels": input_ids[1:]})

    return packed_ds

def save_processed_dataset(ds, name="processed_dataset"):
    file_path = 'alpaca_dataset/{0}.json'.format(name)
    with open(file_path, 'w') as f:
        #for entry in ds:
        json.dump(ds, f)
            #json.dump(entry, f)
            #f.write('\n')

packed_ds = pack_dataset(dataset, tokenizer, 512)
#packed_ds = pack_dataset(dataset, tokenizer, 2048)
#packed_ds = pack_dataset(dataset, tokenizer, 4096)
print("=====================================")
#print(packed_ds)
#save_processed_dataset(packed_ds, name='processed_dataset_small_4096')
#save_processed_dataset(packed_ds, name='processed_dataset')

print("=====================================")
#pytorch_total_params = sum(p.numel() for p in model.parameters())
#print(pytorch_total_params)
#pytorch_total_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(pytorch_total_params_grad)
#lora.mark_only_lora_as_trainable(model)
#pytorch_total_params_grad_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(pytorch_total_params_grad_lora)

