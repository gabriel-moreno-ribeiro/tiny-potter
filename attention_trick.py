import torch
from torch.nn import functional as F

# toy example: we want x[b, t] to become the mean of x[b, :t+1]
torch.manual_seed(1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)

xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)
print(x[0])
print(xbow[0])

# 3x3 example: a lower triangular a with rows summing to 1 makes a @ b the running mean of b
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print('a=', a, 'b=', b, 'c=', c, sep='\n')

wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x  # (T,T) @ (B,T,C) -> (B,T,C)
assert torch.allclose(xbow, xbow2)

tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))  # a token can't look at the future
wei = F.softmax(wei, dim=-1)
print(wei)
xbow3 = wei @ x
assert torch.allclose(xbow, xbow3)

print("all three versions give the same result")
