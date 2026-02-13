import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- LOAD SAME MODEL CODE -----

text = open("input.txt").read()

chars = sorted(list(set(text)))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}

def encode(s): return [stoi[c] for c in s if c in stoi]
def decode(l): return ''.join([itos[i] for i in l])

block_size = 32
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

# ---------------- CHAT MEMORY ----------------

conversation = []

def generate(prompt, max_tokens=200):
    idx = torch.tensor([encode(prompt)], dtype=torch.long)

    for _ in range(max_tokens):
        logits = model(idx)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next = torch.multinomial(probs, 1)
        idx = torch.cat((idx, next), dim=1)

    return decode(idx[0].tolist())

print("\nYour Personal GPT (type 'exit' to quit)\n")

while True:
    user = input("You: ")

    if user == "exit":
        break

    conversation.append(f"User: {user}\nBot:")

    prompt = "".join(conversation)[-500:]

    reply = generate(prompt)

    bot_reply = reply.split("Bot:")[-1]

    conversation.append(bot_reply + "\n")

    print("Bot:", bot_reply.strip())