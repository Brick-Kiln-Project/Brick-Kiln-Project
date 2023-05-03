import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
from torchvision.models import resnet18
import sys
sys.path.append("../Configs/")
import constants

class Resnet(torch.nn.Module):
    def __init__(self, num_channels,constants, image_width=constants.GEE_IMAGE_SHAPE, device=None, pretrained=False):
        super(Resnet, self).__init__()
        self.device = device        
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.conv1 = torch.nn.Conv2d(num_channels, image_width, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 256).to(device)
        self.final_fc = torch.nn.Linear(256, 1).to(device)
        
        self.resnet = self.resnet.to(device)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        resnet_output = self.resnet(x)
        outputs = self.final_fc(resnet_output)
        outputs = self.sigmoid(outputs)
        return outputs
    
    
def load_checkpoint(checkpoint, model, device, optimizer=None):
    """
    Loads a pretrained checkpoint to continue training
    model_checkpoint: Path of the model_checkpoint that ends with .pth
    model: model to load to
    device: devide to load on (gpu/cpu)
    optimizer (optional): optimize to load state to
    """
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    return model


def load_model_from_checkpoint(path, constants, device='cpu'):
    logdir=constants.MODEL_ROOT+path+'_50_training_steps/checkpoints/best_dl_best.pth'
    model = Resnet(3,constants, device=device)
    model = load_checkpoint(logdir, model, device)

    return model

    
if __name__ == "__main__":
    pass