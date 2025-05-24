import torch
import torch.nn as nn
import torch.nn.functional as F

class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))  # 可训练参数

    def forward(self, x):
        delta = self.scale * x
        out = x + delta
        return out

net = TestNet().cuda()
x = torch.randn(1, 1, 4, 4).cuda().requires_grad_()

y = net(x)
loss = y.mean()
loss.backward()

print("scale grad:", net.scale.grad)  # ✅ 应该不是 None
