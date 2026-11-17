import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import tokenizer_compare as tc

COLORS = {'my tokenizer': '#999999', 'K2-Horizon-0.9B': '#56B4E9', 'K2-Horizon-375B-A23B': '#0072B2'}
K2 = [label for label in tc.counters if label != 'my tokenizer']

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
})

def new_chart(ncols=1):
    return plt.subplots(1, ncols, figsize=(8, 4.5), layout='constrained', sharey=True)

def save(fig, name):
    os.makedirs('charts', exist_ok=True)
    fig.savefig(f'charts/{name}.png', dpi=200)  # 8 inches at 200 dpi is 1600 px wide
    plt.close(fig)

def grouped_bars(ax, groups, series, fmt):
    # series maps a tokenizer to one value per group, with None where it can't encode the text
    width = 0.8 / len(series)
    for i, (label, values) in enumerate(series.items()):
        xs = [g + (i - (len(series) - 1) / 2) * width for g in range(len(groups))]
        shown = [(x, v) for x, v in zip(xs, values) if v is not None]
        bars = ax.bar([x for x, _ in shown], [v for _, v in shown], width, color=COLORS[label], label=label)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=9)
        for x, v in zip(xs, values):
            if v is None:
                ax.text(x, 0, 'n/a', ha='center', va='bottom', fontsize=9, color='#555555')
    ax.set_xticks(range(len(groups)), groups)
    ax.margins(y=0.15)

def input_chars_per_token():
    fig, ax = new_chart()
    labels = list(tc.counters)
    bars = ax.bar(labels, [len(tc.text) / count(tc.text) for count in tc.counters.values()], 0.6, color=[COLORS[l] for l in labels])
    ax.bar_label(bars, fmt='%.2f', padding=2)
    ax.margins(y=0.15)
    ax.set_title('Characters per token on input.txt')
    ax.set_ylabel('characters per token')
    save(fig, 'input_chars_per_token')

def flores_ratios():
    sentences = tc.flores_sentences()
    series = {}
    for label in K2:
        english = tc.total(tc.counters[label], sentences['English'])
        series[label] = [tc.total(tc.counters[label], lines) / english for lines in sentences.values()]
    fig, ax = new_chart()
    grouped_bars(ax, list(sentences), series, '%.2f')
    ax.axhline(1.0, color='#555555', linestyle='--', linewidth=1, label='English = 1.0')
    ax.set_title('Tokens relative to English, 200 FLORES-200 sentences')
    ax.set_ylabel('tokens / English tokens')
    ax.legend(frameon=False, loc='upper left', ncols=3)
    save(fig, 'flores_ratio_to_english')

def formats():
    fig, axes = new_chart(2)
    names = [name for name, _, _ in tc.FORMATS]
    xs = range(len(names))
    for ax, label in zip(axes, K2):
        count = tc.counters[label]
        brazil = ax.bar([x - 0.2 for x in xs], [count(br) for _, br, _ in tc.FORMATS], 0.4, color=COLORS[label], label='Brazil')
        us = ax.bar([x + 0.2 for x in xs], [count(us) for _, _, us in tc.FORMATS], 0.4, color='white', edgecolor=COLORS[label], hatch='//', label='US')
        for bars in (brazil, us):
            ax.bar_label(bars, padding=2, fontsize=9)
        ax.set_xticks(list(xs), names)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.margins(y=0.15)
        ax.set_title(label)
        ax.legend(frameon=False)
    axes[0].set_ylabel('tokens')
    fig.suptitle('Brazilian and US formats')
    save(fig, 'formats_brazil_vs_us')

def name_tokens(names, title, name):
    series = {label: [None if split(n) is None else len(split(n)) for n in names] for label, split in tc.splitters.items()}
    fig, ax = new_chart()
    grouped_bars(ax, names, series, '%d')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    ax.set_ylabel('tokens')
    ax.legend(frameon=False)
    save(fig, name)

def code_and_math():
    texts = {'Python function': tc.CODE, 'LaTeX formula': tc.LATEX}
    series = {label: [None if count(t) is None else len(t) / count(t) for t in texts.values()] for label, count in tc.counters.items()}
    fig, ax = new_chart()
    grouped_bars(ax, list(texts), series, '%.2f')
    ax.set_title('Characters per token on code and math')
    ax.set_ylabel('characters per token')
    ax.legend(frameon=False)
    save(fig, 'code_and_math_chars_per_token')

input_chars_per_token()
flores_ratios()
formats()
name_tokens(tc.NAMES, 'Tokens per Harry Potter name', 'harry_potter_names')
name_tokens(tc.PLACES, 'Tokens per Brazilian place name', 'brazilian_places')
code_and_math()
