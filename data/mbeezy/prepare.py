import os
import tiktoken
import numpy as np

n = 2

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    raise FileNotFoundError("Input.txt does not exist")

with open(input_file_path, 'r') as f:
    data = f.read()
len_data = len(data)

# get all the elemental atoms that occur in this text
if n == 1:
    tokens = sorted(list(set(data)))
elif n == 2:
    tokens = sorted(
        list(set([a + b for a, b in zip(data[:-1], data[1:])])))
elif n == 3:
    tokens = sorted(list(set([a+b+c for a, b, c in zip(data[::3], data[1::3], data[2::3])])))
elif n == 4:
    tokens = sorted(list(set([a+b+c+d for a, b, c, d in zip(data[::4], data[1::4], data[2::4], data[3::4])])))
else:
    raise NotImplementedError(f"{n=} is not implemented for obtaining tokens")

vocab_size = len(tokens)
print("all the unique characters:", ''.join(tokens))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
ttoi = { t:i for i,t in enumerate(tokens) }
itot = { i:t for i,t in enumerate(tokens) }

if n == 1:
    def encode(s):
        return [ttoi[z] for z in s]
elif n == 2:
    def encode(s):
        return [ttoi[z] for z in [a+b for a, b in zip(s[::2], s[1::2])]]
elif n == 3:
    def encode(s):
        return [ttoi[z] for z in [a+b+c for a, b, c in zip(s[::3], s[1::3], s[2::3])]]
elif n == 4:
    def encode(s):
        return [ttoi[z] for z in [a+b+c+d for a, b, c, d in zip(s[::4], s[1::4], s[2::4], s[3::4])]]
else:
    raise NotImplementedError(f"{n=} is not implemented")

def decode(l):
    return ''.join([itot[i] for i in l]) # decoder: take a list of integers, output a string

train_data = data[:int(len_data*0.9)]
val_data = data[int(len_data*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
