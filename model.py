import torch

class Net(torch.nn.Module):
    # Convolutional Neural Net architecture

    def __init__(self):
        super(Net, self).__init__() # Input 1 * 64 * 64
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)  
        self.fc1   = torch.nn.Linear(14*14*6, 120)  
        self.fc2   = torch.nn.Linear(120, 40)
        self.fc3   = torch.nn.Linear(40, 5)

    def forward(self, x):
        out = torch.nn.functional.relu(self.conv1(x))  # 3 * 60 * 60 
        out = torch.nn.functional.max_pool2d(out, 2)  # 3 * 30 * 30
        out = torch.nn.functional.relu(self.conv2(out))  # 6 * 28 * 28
        out = torch.nn.functional.max_pool2d(out, 2)  # 6 * 14 * 14
        out = out.view(out.size(0), -1) # flatten
        out = torch.nn.functional.relu(self.fc1(out))  # 120
        out = torch.nn.functional.relu(self.fc2(out))  # 40
        out = self.fc3(out)  # 5 classes
        return out
