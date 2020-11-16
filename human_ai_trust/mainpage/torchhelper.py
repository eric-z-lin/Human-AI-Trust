from torchvision.models import resnet18
from torch import nn

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        # self.densenet121 = torchvision.models.densenet121(pretrained=True)
        # self.densenet121 = torchvision.models.alexnet(pretrained=True)
        self.densenet121 = resnet18(pretrained=True)
        num_ftrs = self.densenet121.fc.in_features
        self.densenet121.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )
        for param in self.densenet121.parameters():
            param.requires_grad = True

        for param in self.densenet121.fc.parameters():
            param.requires_grad = True



    def forward(self, x):
        x = self.densenet121(x)
        return x