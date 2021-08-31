from torch import nn


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ProtonetCNN(nn.Module):
    def __init__(self, n_classes, in_channels=3, out_channels=64, hidden_size=64):
        """Standard prototypical network"""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels),
            nn.AdaptiveAvgPool2d(output_size=(5, 5)),
        )

        self.fc = nn.Linear(out_channels * 5 * 5, n_classes, bias=False)

        print("[Model] ProtonetCNN initialized...")

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.out_channels * 5 * 5)
        x = self.fc(x)
        return x
