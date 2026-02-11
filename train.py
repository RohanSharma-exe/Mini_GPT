import torch
import torch.nn as nn
import torch.nn.functional as F

text = open("input.txt").read()

chars = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

block_size = 8
batch_size = 16

def get_batch():
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(chars), 64)
        self.fc = nn.Linear(64, len(chars))

    def forward(self, x):
        x = self.embed(x)
        x = self.fc(x)
        return x

model = MiniGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(2000):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, len(chars)), yb.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, loss.item())

print("Training finished")

context = torch.zeros((1,1), dtype=torch.long)

for _ in range(200):
    logits = model(context)
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next = torch.multinomial(probs, 1)
    context = torch.cat((context, next), dim=1)

print(decode(context[0].tolist()))