import torch

print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())

device = "mps" if torch.backends.mps.is_available() else "cpu"

x = torch.randn(4, 4, device=device)
w = torch.randn(4, 4, device=device)

y = x @ w
print("y shape:", y.shape)
print("y device:", y.device)


print("x => ")
print(x.device)

print("y => ")
print(w.device)

print("z => ")
print(y.device)
