import hashlib
import os
import tarfile
import unicodedata
import urllib.request

from tokenizers import Tokenizer

K2_TOKENIZERS = [
    ('K2-Horizon-0.9B', 'IFM/K2-Horizon-0.9B', 'fc5c9dc7c13df2360acee789fca2cd55d439e644'),
    ('K2-Horizon-375B-A23B', 'IFM/K2-Horizon-375B-A23B', 'bb3bfb4b532488c05e84a090301f8259c68b319c'),
]

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(set(text))
vocab = set(chars)

def k2_tokenizer(repo, revision):
    tok = Tokenizer.from_pretrained(repo, revision=revision)
    ids = lambda s: tok.encode(s, add_special_tokens=False).ids
    # names mostly come after a space in the books, and K2 merges that space into the first piece
    split = lambda name: [p.strip() for p in (tok.decode([i]) for i in ids(' ' + name)) if p.strip()]
    return lambda s: len(ids(s)), split, tok.get_vocab_size()

# my tokenizer is one token per character; None means a character isn't in its vocabulary
fits = lambda s: set(s) <= vocab
counters = {'my tokenizer': lambda s: len(s) if fits(s) else None}
splitters = {'my tokenizer': lambda s: list(s) if fits(s) else None}
vocab_sizes = {'my tokenizer': len(chars)}
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
fmt_pieces = lambda p: 'n/a' if p is None else f"{'·'.join(p)} ({len(p)})"

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

def books_and_sentences():
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

NAMES = ['Harry', 'Hermione', 'Dumbledore', 'Voldemort', 'Hogwarts', 'Quidditch', 'Expelliarmus']

def names():
    table(['name'], [[name] + [fmt_pieces(split(name)) for split in splitters.values()] for name in NAMES])

FLORES_URL = 'https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz'
FLORES_SHA256 = 'b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6'
FLORES_LANGUAGES = {
    'English': 'eng_Latn', 'Portuguese': 'por_Latn', 'Spanish': 'spa_Latn', 'Arabic': 'arb_Arab',
    'Hindi': 'hin_Deva', 'Chinese': 'zho_Hans', 'Indonesian': 'ind_Latn',
}

def languages():
    path = 'flores200_dataset.tar.gz'
    if not os.path.exists(path):
        urllib.request.urlretrieve(FLORES_URL, path)
    with open(path, 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == FLORES_SHA256, f'{path} is not the expected file, delete it and run again'
    with tarfile.open(path) as tar:
        read = lambda code: tar.extractfile(f'./flores200_dataset/devtest/{code}.devtest').read().decode('utf-8')
        sentences = {lang: read(code).splitlines()[:200] for lang, code in FLORES_LANGUAGES.items()}
    english = [total(count, sentences['English']) for count in counters.values()]
    rows = []
    for lang, lines in sentences.items():
        cells = []
        for count, en in zip(counters.values(), english):
            n = total(count, lines)
            cells.append(fmt(n) + ('' if lang == 'English' or None in (n, en) else f' ({n / en:.2f})'))
        rows.append([lang, f'{sum(map(len, lines)):,}'] + cells)
    table(['language', 'characters'], rows)

INFORMAL_FORMAL = [
    ('vc vem hj ou não?', 'Você vem hoje ou não?'),
    ('tô cansadão, vou dormir cedo', 'Estou muito cansado, vou dormir cedo.'),
    ('oxe, que história é essa?', 'Que história é essa?'),
    ('vixe, esqueci a chave em casa', 'Esqueci a chave em casa.'),
    ('esse forró tava arretado demais', 'Esse forró estava muito bom.'),
    ('pq vc não me ligou?', 'Por que você não me ligou?'),
    ('tá bom, a gente se vê amanhã', 'Está bem, nós nos vemos amanhã.'),
    ('blz, te espero lá fora', 'Tudo bem, espero você lá fora.'),
    ('mano, o jogo ontem foi mto doido', 'O jogo de ontem foi muito emocionante.'),
    ('num sei se vou conseguir ir não', 'Não sei se vou conseguir ir.'),
    ('bora tomar um açaí?', 'Vamos tomar um açaí?'),
    ('cê tá doido, é?', 'Você está louco?'),
    ('eita, choveu mto hj', 'Choveu muito hoje.'),
    ('tb quero ir pra praia', 'Também quero ir para a praia.'),
    ('vlw pela ajuda, mano', 'Obrigado pela ajuda.'),
    ('manda no zap q eu vejo dps', 'Mande pelo WhatsApp que eu vejo depois.'),
    ('tô de boa, tá tudo sussa', 'Estou bem, está tudo tranquilo.'),
    ('o rolê de sábado foi top', 'O passeio de sábado foi ótimo.'),
    ('mainha fez cuscuz pro café', 'Minha mãe fez cuscuz para o café da manhã.'),
    ('ngm sabe onde fica isso não', 'Ninguém sabe onde fica isso.'),
    ('tmj, qualquer coisa me chama', 'Conte comigo, se precisar de algo, me chame.'),
    ('sdds de vc, vem me visitar', 'Estou com saudade de você, venha me visitar.'),
    ('fds vou pra casa da minha vó', 'No fim de semana vou para a casa da minha avó.'),
    ('nd a ver isso q vc falou', 'Isso que você falou não tem nada a ver.'),
    ('uai, cê num vai com a gente?', 'Você não vai conosco?'),
    ('bah, tchê, que frio hj', 'Está muito frio hoje.'),
    ('obg, vc é um anjo', 'Obrigado, você é um anjo.'),
    ('qnd vc chegar me avisa', 'Quando você chegar, me avise.'),
    ('cadê o controle da tv?', 'Onde está o controle da televisão?'),
    ('flw, até amanhã', 'Tchau, até amanhã.'),
]
PLACES = ['Salvador', 'Missão Velha', 'Fortaleza', 'Juazeiro do Norte', 'Florianópolis']

def strip_accents(s):
    # NFD splits a letter like ã into a plus a combining mark, so dropping the marks removes the accents
    return unicodedata.normalize('NFC', ''.join(c for c in unicodedata.normalize('NFD', s) if not unicodedata.combining(c)))

def informal_portuguese():
    formal = [f for _, f in INFORMAL_FORMAL]
    variants = {
        'informal': [i for i, _ in INFORMAL_FORMAL],
        'formal': formal,
        'formal without accents': [strip_accents(f) for f in formal],
    }
    rows = [[label, f'{sum(map(len, lines)):,}'] + [fmt(total(count, lines)) for count in counters.values()] for label, lines in variants.items()]
    table([f'{len(INFORMAL_FORMAL)} sentences', 'characters'], rows)
    table(['place'], [[place] + [fmt_pieces(split(place)) for split in splitters.values()] for place in PLACES])

books_and_sentences()
names()
languages()
informal_portuguese()
