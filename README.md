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

These are the values from the end of the lecture, 10.81M parameters in total. The final run ended at train 1.08 / val 1.18.

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

## Credit

The architecture and the order of the steps come from Andrej Karpathy's lecture [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and its repo [karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture).
