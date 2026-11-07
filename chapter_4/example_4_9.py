# Example 4.9: The CNN of Example 4.2, rebuilt in PyTorch
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0); torch.set_num_threads(1)
digits = load_digits()
X = torch.tensor(digits.images / 16.0, dtype=torch.float32).unsqueeze(1)   # (N, 1, 8, 8): one channel
y = torch.tensor(digits.target)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 8 filters of 3x3, "same" padding
            nn.ReLU(),
            nn.MaxPool2d(2))                             # 8x8 -> 4x4
        self.classifier = nn.Linear(8 * 4 * 4, 10)      # softmax is inside the loss function
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

model = SmallCNN()
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)

for epoch in range(1, 41):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()                     # 1) clear the old gradients
        loss = loss_fn(model(xb), yb)       # 2) forward pass and loss
        loss.backward()                     # 3) backpropagation: autograd computes every gradient
        opt.step()                          # 4) update the weights
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            acc_tr = (model(X_tr).argmax(1) == y_tr).float().mean().item()
            acc_te = (model(X_te).argmax(1) == y_te).float().mean().item()
        print(f"epoch {epoch}: train acc = {acc_tr:.3f} | test acc = {acc_te:.3f}")
