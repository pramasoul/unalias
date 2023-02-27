import torch
import torch.nn as nn
import torch.nn.functional as F

class AliasingRectificationModel(nn.Module):
    def __init__(self):
        super(AliasingRectificationModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=9, stride=1, padding=4)
        self.attention = nn.MultiheadAttention(embed_dim=1, num_heads=4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x, _ = self.attention(x, x, x)
        x = F.relu(self.conv3(x))
        return x
