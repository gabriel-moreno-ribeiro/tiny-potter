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
    ('I drink coffee every morning.', 'Eu tomo café todas as manhãs.'),
    ("The train leaves at eight o'clock.", 'O trem sai às oito horas.'),
    ('Please close the door when you leave.', 'Por favor, feche a porta quando sair.'),
    ('He works in a small office downtown.', 'Ele trabalha em um pequeno escritório no centro.'),
    ('Our dog sleeps on the sofa.', 'Nosso cachorro dorme no sofá.'),
    ('It rained all night long.', 'Choveu a noite inteira.'),
    ('The book is on the kitchen table.', 'O livro está na mesa da cozinha.'),
    ('They bought a new car last week.', 'Eles compraram um carro novo na semana passada.'),
    ("I don't understand this question.", 'Eu não entendo esta pergunta.'),
    ('Can you help me carry these boxes?', 'Você pode me ajudar a carregar estas caixas?'),
    ('The store closes at nine at night.', 'A loja fecha às nove da noite.'),
    ('My grandmother makes the best cake.', 'Minha avó faz o melhor bolo.'),
    ('We watched a movie on Saturday.', 'Nós assistimos a um filme no sábado.'),
    ('The water in the lake is very clean.', 'A água do lago é muito limpa.'),
    ('She speaks three languages.', 'Ela fala três idiomas.'),
    ('I forgot my umbrella at home.', 'Eu esqueci meu guarda-chuva em casa.'),
    ('The museum is free on Sundays.', 'O museu é gratuito aos domingos.'),
    ('He runs in the park every evening.', 'Ele corre no parque todas as noites.'),
    ('The soup is too hot to eat.', 'A sopa está quente demais para comer.'),
    ('Where is the nearest hospital?', 'Onde fica o hospital mais próximo?'),
    ('My sister is studying to be a doctor.', 'Minha irmã está estudando para ser médica.'),
    ('The birds sing early in the morning.', 'Os pássaros cantam cedo de manhã.'),
    ('We need to buy bread and milk.', 'Precisamos comprar pão e leite.'),
    ('The meeting was very long and boring.', 'A reunião foi muito longa e chata.'),
    ('I will call you tomorrow afternoon.', 'Vou te ligar amanhã à tarde.'),
    ('The teacher explained the lesson again.', 'A professora explicou a lição de novo.'),
    ('This road goes to the beach.', 'Esta estrada vai até a praia.'),
    ('He lost his wallet on the bus.', 'Ele perdeu a carteira no ônibus.'),
    ('The flowers in the garden are beautiful.', 'As flores do jardim são lindas.'),
    ('I like to read before going to sleep.', 'Eu gosto de ler antes de dormir.'),
    ('The plane arrived two hours late.', 'O avião chegou com duas horas de atraso.'),
    ('Our neighbors are very friendly.', 'Nossos vizinhos são muito simpáticos.'),
    ('She painted the walls of her room blue.', 'Ela pintou as paredes do quarto dela de azul.'),
    ('The baby is sleeping in the other room.', 'O bebê está dormindo no outro quarto.'),
    ('We walked along the river for an hour.', 'Nós caminhamos ao longo do rio por uma hora.'),
    ('The price of fruit went up this month.', 'O preço das frutas subiu este mês.'),
    ('He always forgets my birthday.', 'Ele sempre esquece o meu aniversário.'),
    ('The library opens at ten in the morning.', 'A biblioteca abre às dez da manhã.'),
    ('I have never seen the ocean.', 'Eu nunca vi o mar.'),
    ('The cat jumped onto the roof.', 'O gato pulou para o telhado.'),
    ('My parents live in a small town.', 'Meus pais moram em uma cidade pequena.'),
    ('The students are taking an exam today.', 'Os alunos estão fazendo uma prova hoje.'),
    ('Turn left at the next corner.', 'Vire à esquerda na próxima esquina.'),
    ('The kitchen smells of fresh bread.', 'A cozinha está com cheiro de pão fresco.'),
    ('We are very tired after the long trip.', 'Estamos muito cansados depois da longa viagem.'),
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

vocab = set(chars)
# my tokenizer is one token per character; None means a character isn't in its vocabulary
my_count = lambda s: len(s) if set(s) <= vocab else None
counters, splitters, vocab_sizes = {'my tokenizer': my_count}, {'my tokenizer': list}, {'my tokenizer': len(chars)}
for label, repo, revision in K2_TOKENIZERS:
    counters[label], splitters[label], vocab_sizes[label] = k2_tokenizer(repo, revision)

def table(lead, rows):
    header = lead + list(counters)
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')
    print()

def total(count, strings):
    counts = [count(s) for s in strings]
    return None if None in counts else sum(counts)

fmt = lambda n: 'n/a' if n is None else f'{n:,}'
counts = [count(text) for count in counters.values()]
en = [total(count, [e for e, _ in SENTENCES]) for count in counters.values()]
pt = [total(count, [p for _, p in SENTENCES]) for count in counters.values()]
table([''], [
    ['vocabulary size'] + [f'{v:,}' for v in vocab_sizes.values()],
    ['tokens in input.txt'] + [f'{n:,}' for n in counts],
    ['characters per token'] + [f'{len(text) / n:.2f}' for n in counts],
    [f'tokens for {len(SENTENCES)} English sentences'] + [fmt(n) for n in en],
    ['tokens for their Portuguese translations'] + [fmt(n) for n in pt],
    ['Portuguese/English tokens'] + ['n/a' if None in (e, p) else f'{p / e:.2f}' for e, p in zip(en, pt)],
])
missing = sorted(set(''.join(p for _, p in SENTENCES)) - vocab)
print(f"n/a: my tokenizer only has the {len(chars)} characters of input.txt, so it can't encode {' '.join(missing)}\n")

rows = []
for name in NAMES:
    pieces = [split(name) for split in splitters.values()]
    rows.append([name] + [f"{'·'.join(p)} ({len(p)})" for p in pieces])
table(['name'], rows)
