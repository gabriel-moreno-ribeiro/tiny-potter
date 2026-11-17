import sys
import torch
from gpt import GPTLanguageModel, encode, decode, stoi, device

prompt = ' '.join(sys.argv[1:])
if not prompt:
    sys.exit('usage: python sample.py "Harry looked at"')
unknown = set(prompt) - set(stoi)
if unknown:
    sys.exit(f"not in the vocabulary: {''.join(sorted(unknown))}")

model = GPTLanguageModel().to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
with torch.no_grad():
    print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))
