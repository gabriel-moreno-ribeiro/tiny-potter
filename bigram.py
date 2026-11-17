import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters:", len(text))
print(text[:500])

# all the unique characters in the text are the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# map characters to integers and back
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # list of integers -> string

sentence = "Mischief managed."
print(encode(sentence))
print(decode(encode(sentence)))
assert decode(encode(sentence)) == sentence

# the whole text as one long tensor of integers
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# first 90% is train, the rest is validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(len(train_data), len(val_data))

torch.manual_seed(1337)
batch_size = 4  # how many independent sequences we process in parallel
block_size = 8  # maximum context length for predictions

# a chunk of block_size+1 characters packs block_size examples
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context.tolist()} the target: {target}")

def get_batch(split):
    # random chunks of inputs x and their targets y (x shifted by one)
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:', xb.shape)
print(xb)
print('targets:', yb.shape)
print(yb)
