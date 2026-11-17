# LLM from scratch (tiny potter)

I built a small character-level GPT from scratch in PyTorch and trained it on the Harry Potter books. The code follows Andrej Karpathy's lecture in the same order: a bigram model first, then self-attention, multi-head attention, feed-forward layers, transformer blocks, residual connections and layernorm, and last a scaled-up model with dropout. The git log has one commit per step.

## Data

`input.txt` is not included in this repo. You have to supply it yourself: a plain text file called `input.txt` in the same folder as the scripts. I used the Harry Potter books, 6,304,282 characters of plain ASCII with 90 distinct characters. The tokenizer works on single characters, so those 90 characters are the whole vocabulary. The first 90% of the text is the training set and the last 10% is the validation set.

## How to run

You only need Python 3 and PyTorch.

```
pip install torch
python bigram.py                    # bigram baseline
python attention_trick.py           # the averaging trick behind self-attention
python gpt.py                       # trains the GPT and saves model.pt
python sample.py "Harry looked at"  # continues a prompt with the saved model
```

`gpt.py` uses CUDA if it finds it, then Apple's MPS, then the CPU. On my M5 Pro the full training run took about 33 minutes on MPS. `sample.py` imports the model from `gpt.py`, so it needs the same `input.txt` to rebuild the vocabulary.

## Final hyperparameters

| Hyperparameter | Value |
|---|---|
| batch_size | 64 |
| block_size | 256 |
| max_iters | 5000 |
| learning_rate | 3e-4 |
| n_embd | 384 |
| n_head | 6 |
| n_layer | 6 |
| dropout | 0.2 |
| eval_interval | 500 |
| eval_iters | 200 |

These are the values from the end of the lecture. The model has 10.808154M parameters, and the final run ended at train 1.08 / val 1.18.

## Validation loss at each stage

| Stage | Val loss |
|---|---|
| bigram | 2.45 |
| one self-attention head | 2.39 |
| multi-head attention (4 heads of 8) | 2.28 |
| feed-forward layer | 2.24 |
| 3 blocks, no residuals | 2.36 |
| residual connections and projections | 1.99 |
| layernorm | 1.96 |
| final (scaled up, dropout) | 1.18 |

Every row except the last is a small model trained on the CPU with block_size 8 and batch_size 32, like the middle part of the lecture. Stacking blocks made the loss worse until I added the residual connections.

## Sample

`python sample.py "Harry looked at"` prints the prompt followed by 500 generated characters:

```
Harry looked at him; he was now waving in his carriage as though he'd never been absensive ink half empty Weasleys', he's got to ask . . ." bowed him; and he was not sure he had a suggested loudly that it had his under which he was sitting himself about to buy the Decree Against the Dark Lord say, that is it back? She was cooling, half-coldly most of school phitles with her can.

Hermione was surely smoothly going how ahead as though he had to rise the Daily, he had an unbreath look out at Leanne.

Uncle Verno
```

## Comparing with K2's tokenizer

`tokenizer_compare.py` compares my character tokenizer with the tokenizers of MBZUAI's [K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B) and [K2-Horizon-375B-A23B](https://huggingface.co/IFM/K2-Horizon-375B-A23B). Run it with `pip install tokenizers` and `python tokenizer_compare.py`; it downloads only each model's `tokenizer.json`. The two K2 tokenizers are different: the 375B-A23B one has 250,624 tokens instead of 64,256, and none of the tokenizer files have the same hash. The Portuguese/English ratio comes from 50 simple English sentences and their Portuguese translations, which are in the script.

|  | my tokenizer | K2-Horizon-0.9B | K2-Horizon-375B-A23B |
|---|---|---|---|
| vocabulary size | 90 | 64,256 | 250,624 |
| tokens in input.txt | 6,304,282 | 1,613,313 | 1,472,117 |
| characters per token | 1.00 | 3.91 | 4.28 |
| tokens for 50 English sentences | 1,682 | 390 | 383 |
| tokens for their Portuguese translations | n/a | 568 | 441 |
| Portuguese/English tokens | n/a | 1.46 | 1.15 |

My tokenizer gets n/a for Portuguese because its vocabulary is only the 90 plain ASCII characters of input.txt, so it can't encode à, á, ã, ç, é, ê, ó or ô.

K2 splits the names the way they appear in the books, after a space.

| name | my tokenizer | K2-Horizon-0.9B | K2-Horizon-375B-A23B |
|---|---|---|---|
| Harry | H·a·r·r·y (5) | Harry (1) | Harry (1) |
| Hermione | H·e·r·m·i·o·n·e (8) | Herm·ione (2) | Hermione (1) |
| Dumbledore | D·u·m·b·l·e·d·o·r·e (10) | D·umbled·ore (3) | D·umbled·ore (3) |
| Voldemort | V·o·l·d·e·m·o·r·t (9) | V·old·em·ort (4) | V·old·em·ort (4) |
| Hogwarts | H·o·g·w·a·r·t·s (8) | H·og·w·arts (4) | Hogwarts (1) |
| Quidditch | Q·u·i·d·d·i·t·c·h (9) | Qu·idd·itch (3) | Qu·idd·itch (3) |
| Expelliarmus | E·x·p·e·l·l·i·a·r·m·u·s (12) | Exp·elli·arm·us (4) | Exp·elli·arm·us (4) |

Both K2 tokenizers need about a quarter of the tokens for the books, and the 375B-A23B one keeps Harry, Hermione and Hogwarts whole where the 0.9B keeps only Harry. They cost about the same on the English sentences, but Portuguese takes 46% more tokens than English with the 0.9B and 15% more with the 375B-A23B.

For the language test I took the first 200 sentences of the [FLORES-200](https://github.com/facebookresearch/flores) devtest set, which has the same sentences in every language. The script downloads Meta's archive (CC BY-SA 4.0) and caches it as `flores200_dataset.tar.gz`, so none of the text is in this repo. Each K2 cell shows the tokens and, in parentheses, the ratio to English.

| language | characters | my tokenizer | K2-Horizon-0.9B | K2-Horizon-375B-A23B |
|---|---|---|---|---|
| English | 26,763 | n/a | 5,871 | 5,396 |
| Portuguese | 29,300 | n/a | 8,380 (1.43) | 6,816 (1.26) |
| Spanish | 31,773 | n/a | 8,905 (1.52) | 7,314 (1.36) |
| Arabic | 23,571 | n/a | 6,826 (1.16) | 5,693 (1.06) |
| Hindi | 26,550 | n/a | 9,392 (1.60) | 7,390 (1.37) |
| Chinese | 9,049 | n/a | 8,040 (1.37) | 6,285 (1.16) |
| Indonesian | 29,092 | n/a | 7,832 (1.33) | 6,015 (1.11) |

The 375B-A23B needs fewer tokens than the 0.9B in every language; its ratios to English go from 1.06 to 1.37, the 0.9B's from 1.16 to 1.60. Arabic is the cheapest language after English for both; the 0.9B's tokenizer manifest says it was built from a 64k Jais tokenizer (Jais is an Arabic-English model). My tokenizer gets n/a even for English, because 11 of the 200 sentences contain characters like —, $, ’ or ö.

The informal Portuguese test has 30 sentences with slang and abbreviations (vc, tá, pq, oxe, vixe, arretado and others), the same 30 in formal Portuguese, and the formal ones with the accents removed.

| 30 sentences | characters | my tokenizer | K2-Horizon-0.9B | K2-Horizon-375B-A23B |
|---|---|---|---|---|
| informal | 715 | n/a | 272 | 233 |
| formal | 877 | n/a | 293 | 238 |
| formal without accents | 877 | 877 | 296 | 249 |

Brazilian place names, split the same way as the names above:

| place | my tokenizer | K2-Horizon-0.9B | K2-Horizon-375B-A23B |
|---|---|---|---|
| Salvador | S·a·l·v·a·d·o·r (8) | Salvador (1) | Salvador (1) |
| Missão Velha | n/a | Miss·ão·Vel·ha (4) | Miss·ão·Vel·ha (4) |
| Fortaleza | F·o·r·t·a·l·e·z·a (9) | Fort·ale·za (3) | Fortaleza (1) |
| Juazeiro do Norte | J·u·a·z·e·i·r·o· ·d·o· ·N·o·r·t·e (17) | Ju·aze·iro·do·Norte (5) | Ju·aze·iro·do·Norte (5) |
| Florianópolis | n/a | Flor·ian·ó·pol·is (5) | Florian·ópolis (2) |

The informal sentences have 18% fewer characters than the formal ones but only 7% fewer tokens with the 0.9B and 2% fewer with the 375B-A23B. Removing the accents makes both K2 tokenizers slightly worse (296 and 249 tokens instead of 293 and 238), and the version without accents is the only one my tokenizer can read. Both keep Salvador whole, and the 375B-A23B also keeps Fortaleza and cuts Florianópolis into two pieces instead of five; Missão Velha and Juazeiro do Norte break up the same way in both.

The format test compares Brazilian and US money, dates, phone numbers and postal codes; each cell is Brazil / US tokens.

| format | Brazil | US | my tokenizer | K2-Horizon-0.9B | K2-Horizon-375B-A23B |
|---|---|---|---|---|---|
| money | `R$ 1.234,56` | `$1,234.56` | n/a / n/a | 8 / 6 | 8 / 6 |
| date | `17/11/2026` | `11/17/2026` | 10 / 10 | 6 / 6 | 6 / 6 |
| phone | `(71) 99999-9999` | `(555) 123-4567` | 15 / 14 | 9 / 8 | 9 / 8 |
| postal code | `40140-650` | `90210` | 9 / 5 | 4 / 2 | 4 / 2 |

Both K2 tokenizers split these exactly the same way: numbers are cut into chunks of up to three digits from the left, so 2026 becomes 202 and 6, and 99999 becomes 999 and 99. The Brazilian money format costs two extra tokens because R and the space before the number are tokens of their own, and a CEP takes 4 tokens against 2 for a ZIP code. My tokenizer can't do money at all because $ isn't in its vocabulary.

## Credit

The architecture and the order of the steps come from Andrej Karpathy's lecture [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and its repo [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture).
