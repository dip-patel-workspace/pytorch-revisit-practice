import torch
import torch.nn as nn

# Note:
# Calculate Shape of Layer: # Output Size = floor((Input Size - Kernel Size + 2 × Padding) / Stride) + 1

# LeNet Implementation:
class LeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, 
                            kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120,
                            kernel_size=5, stride=1, padding=0)
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.linear2 = nn.Linear(in_features=84, out_features=10) # Output Layer
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        print(f"Weight Shape: {self.conv1.weight.shape} | Bias Shape: {self.conv1.bias.shape}")
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)   # convert (no of examples, 120, 1, 1) into (no of examples, 120)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Eexcute
def run_sample_input():
    x = torch.randn(64, 1, 32, 32)
    model = LeNet()
    return model(x)

#
if __name__ == "__main__":
    output = run_sample_input()
    print(output.shape)