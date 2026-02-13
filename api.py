import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# ----- Load model -----

text = open("input.txt").read()

chars = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}

def encode(s): return [stoi[c] for c in s if c in stoi]
def decode(l): return ''.join([itos[i] for i in l])

embed_size = 64

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.key = nn.Linear(embed_size, embed_size)
        self.query = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        w = q @ k.transpose(-2, -1)
        w = F.softmax(w, dim=-1)
        return w @ v

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4, embed_size)
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ff(x)
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(chars), embed_size)
        self.blocks = nn.Sequential(Block(), Block(), Block())
        self.fc = nn.Linear(embed_size, len(chars))

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return self.fc(x)

model = GPT()
model.load_state_dict(torch.load("model.pt"))
model.eval()

conversation = []

def generate(prompt, tokens=200, temperature=0.8, top_k=20):
    idx = torch.tensor([encode(prompt)], dtype=torch.long)

    for _ in range(tokens):
        logits = model(idx)[:, -1, :] / temperature

        v, ix = torch.topk(logits, top_k)
        probs = F.softmax(v, dim=-1)

        next = ix[0][torch.multinomial(probs, 1)]
        idx = torch.cat((idx, next.view(1,1)), dim=1)

    return decode(idx[0].tolist())

class Msg(BaseModel):
    message: str

@app.post("/chat")
def chat(msg: Msg):
    conversation.append(f"User: {msg.message}\nAssistant:")
    prompt = "".join(conversation)[-800:]
    out = generate(prompt)
    reply = out.split("Assistant:")[-1]
    conversation.append(reply + "\n")
    return {"reply": reply.strip()}