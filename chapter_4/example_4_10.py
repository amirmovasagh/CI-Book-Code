# Example 4.10: A small U-Net in PyTorch for segmenting the synthetic cells of Example 4.7
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

rng = np.random.default_rng(1); torch.manual_seed(0); torch.set_num_threads(1)

def make_image(n=96, n_cells=7):                 # the same generator as in Example 4.7
    yy, xx = np.mgrid[:n, :n]
    mask = np.zeros((n, n), bool)
    for _ in range(n_cells):
        cy, cx, r = rng.uniform(10, n - 10, 2).tolist() + [rng.uniform(5, 11)]
        mask |= (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    img = 0.35 + 0.35 * mask + ndi.gaussian_filter(rng.normal(0, 0.25, (n, n)), 1)
    img += 0.15 * np.sin(xx / 15)
    return img, mask

def to_tensor(pairs):
    imgs, masks = zip(*pairs)
    return (torch.tensor(np.array(imgs), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(masks), dtype=torch.float32).unsqueeze(1))

X_tr, Y_tr = to_tensor([make_image() for _ in range(40)])
X_te, Y_te = to_tensor([make_image() for _ in range(10)])

def block(c_in, c_out):                           # two 3x3 convolutions with ReLU
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU())

class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = block(1, 8), block(8, 16)      # encoder (contracting path)
        self.pool = nn.MaxPool2d(2)
        self.bottom = block(16, 32)
        self.up2, self.dec2 = nn.ConvTranspose2d(32, 16, 2, stride=2), block(32, 16)   # decoder
        self.up1, self.dec1 = nn.ConvTranspose2d(16, 8, 2, stride=2), block(16, 8)
        self.out = nn.Conv2d(8, 1, 1)                          # one logit per pixel
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1)); b = self.bottom(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1))       # skip connection
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))      # skip connection
        return self.out(d1)

def dice(pred, true):
    return (2 * (pred * true).sum() / (pred.sum() + true.sum())).item()

model = TinyUNet()
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.Adam(model.parameters(), lr=0.003)
loss_fn = nn.BCEWithLogitsLoss()
losses = []
for epoch in range(1, 61):
    model.train()
    for k in range(0, 40, 8):                                  # mini-batches of 8 images
        opt.zero_grad()
        loss = loss_fn(model(X_tr[k:k + 8]), Y_tr[k:k + 8]); loss.backward(); opt.step()
    losses.append(loss.item())
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            pred = (model(X_te) > 0).float()
        print(f"epoch {epoch}: loss = {loss.item():.4f} | test Dice = {dice(pred, Y_te):.3f}")

thresh = (X_te > X_te.mean(dim=(2, 3), keepdim=True)).float()  # baseline of Example 4.7
print(f"global threshold: test Dice = {dice(thresh, Y_te):.3f}")

fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
for a, (im, t) in zip(ax[:3], [(X_te[0, 0], "Input image"), (Y_te[0, 0], "Ground truth"), (pred[0, 0], "U-Net prediction")]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
ax[3].plot(range(1, 61), losses); ax[3].set_xlabel("epoch"); ax[3].set_ylabel("training loss"); ax[3].set_title("Training loss")
plt.tight_layout(); plt.savefig("fig4_17.png", dpi=300)
