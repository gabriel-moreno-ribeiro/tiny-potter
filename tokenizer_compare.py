from tokenizers import Tokenizer

K2_REPO = 'IFM/K2-Horizon-0.9B'
K2_REVISION = 'fc5c9dc7c13df2360acee789fca2cd55d439e644'
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
chars = sorted(set(text))  # my tokenizer is one token per character, so counts are just len()

k2 = Tokenizer.from_pretrained(K2_REPO, revision=K2_REVISION)
k2_ids = lambda s: k2.encode(s, add_special_tokens=False).ids

n_k2 = len(k2_ids(text))
en_mine = sum(len(en) for en, _ in SENTENCES)
pt_mine = sum(len(pt) for _, pt in SENTENCES)
en_k2 = sum(len(k2_ids(en)) for en, _ in SENTENCES)
pt_k2 = sum(len(k2_ids(pt)) for _, pt in SENTENCES)

print('| | my tokenizer | K2-Horizon-0.9B |')
print('|---|---|---|')
print(f'| vocabulary size | {len(chars)} | {k2.get_vocab_size():,} |')
print(f'| tokens in input.txt | {len(text):,} | {n_k2:,} |')
print(f'| characters per token | 1.00 | {len(text) / n_k2:.2f} |')
print(f'| Portuguese/English tokens | {pt_mine / en_mine:.2f} | {pt_k2 / en_k2:.2f} |')

print('\n| name | my tokenizer | K2-Horizon-0.9B |')
print('|---|---|---|')
for name in NAMES:
    # names mostly come after a space in the books, and K2 merges that space into the first piece
    pieces = [p.strip() for p in (k2.decode([i]) for i in k2_ids(' ' + name)) if p.strip()]
    print(f"| {name} | {'·'.join(name)} ({len(name)}) | {'·'.join(pieces)} ({len(pieces)}) |")

print('\n| English | Portuguese | my EN | my PT | K2 EN | K2 PT |')
print('|---|---|---|---|---|---|')
for en, pt in SENTENCES:
    print(f'| {en} | {pt} | {len(en)} | {len(pt)} | {len(k2_ids(en))} | {len(k2_ids(pt))} |')
print(f'| total | | {en_mine} | {pt_mine} | {en_k2} | {pt_k2} |')

missing = sorted(set(''.join(pt for _, pt in SENTENCES)) - set(chars))
print(f"\nPortuguese characters my tokenizer doesn't have: {' '.join(missing)}")
