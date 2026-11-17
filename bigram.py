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
