import torch.nn as nn

class WeatherNN(nn.Module):
    def __init__(self):
        super(WeatherNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x)
