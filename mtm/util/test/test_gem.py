import torch

from mtm.util.util import GeM


t1 = torch.rand([1, 64, 8, 8])
t2 = t1.mean(dim=[-1, -2])
gem = GeM()
print(gem.p)

optimizer = torch.optim.SGD(gem.parameters(), lr=0.01, momentum=0.9)

for i in range(100):
    optimizer.zero_grad()

    outputs = gem(t1).squeeze(-1).squeeze(-1)
    loss = torch.norm(t2 - outputs, 2)
    loss.backward()
    optimizer.step()

print(gem.p)
