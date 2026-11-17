from tokenizers import Tokenizer

K2_TOKENIZERS = [
    ('K2-Horizon-0.9B', 'IFM/K2-Horizon-0.9B', 'fc5c9dc7c13df2360acee789fca2cd55d439e644'),
    ('K2-Horizon-375B-A23B', 'IFM/K2-Horizon-375B-A23B', 'bb3bfb4b532488c05e84a090301f8259c68b319c'),
]
NAMES = ['Harry', 'Hermione', 'Dumbledore', 'Voldemort', 'Hogwarts', 'Quidditch', 'Expelliarmus']
SENTENCES = [
    ('The weather is very cold today.', 'O tempo está muito frio hoje.'),
    ('She opened the window and looked outside.', 'Ela abriu a janela e olhou para fora.'),
    ('We are going to the market after lunch.', 'Nós vamos ao mercado depois do almoço.'),
    ('My brother cannot find his keys.', 'Meu irmão não consegue encontrar as chaves dele.'),
    ('The children were happy to see the snow.', 'As crianças ficaram felizes ao ver a neve.'),
]

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))

def k2_tokenizer(repo, revision):
    tok = Tokenizer.from_pretrained(repo, revision=revision)
    ids = lambda s: tok.encode(s, add_special_tokens=False).ids
    # names mostly come after a space in the books, and K2 merges that space into the first piece
    split = lambda name: [p.strip() for p in (tok.decode([i]) for i in ids(' ' + name)) if p.strip()]
    return lambda s: len(ids(s)), split, tok.get_vocab_size()

# my tokenizer is one token per character, so counting is len() and splitting is list()
counters, splitters, vocab_sizes = {'my tokenizer': len}, {'my tokenizer': list}, {'my tokenizer': len(chars)}
for label, repo, revision in K2_TOKENIZERS:
    counters[label], splitters[label], vocab_sizes[label] = k2_tokenizer(repo, revision)

def table(lead, rows):
    header = lead + list(counters)
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')
    print()

counts = [count(text) for count in counters.values()]
en = [sum(count(e) for e, _ in SENTENCES) for count in counters.values()]
pt = [sum(count(p) for _, p in SENTENCES) for count in counters.values()]
table([''], [
    ['vocabulary size'] + [f'{v:,}' for v in vocab_sizes.values()],
    ['tokens in input.txt'] + [f'{n:,}' for n in counts],
    ['characters per token'] + [f'{len(text) / n:.2f}' for n in counts],
    ['Portuguese/English tokens'] + [f'{p / e:.2f}' for e, p in zip(en, pt)],
])

rows = []
for name in NAMES:
    pieces = [split(name) for split in splitters.values()]
    rows.append([name] + [f"{'·'.join(p)} ({len(p)})" for p in pieces])
table(['name'], rows)

rows = [[e, p] + [f'{count(e)} / {count(p)}' for count in counters.values()] for e, p in SENTENCES]
table(['English', 'Portuguese'], rows + [['total', ''] + [f'{a} / {b}' for a, b in zip(en, pt)]])

missing = sorted(set(''.join(p for _, p in SENTENCES)) - set(chars))
print(f"Portuguese characters my tokenizer doesn't have: {' '.join(missing)}")
