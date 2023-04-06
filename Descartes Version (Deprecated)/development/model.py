import torch
import tempfile
import descarteslabs as dl
from torchvision.models import resnet18
import sys
sys.path.append("../UI_Labeling")
import config

class Resnet(torch.nn.Module):
    def __init__(self, num_channels, image_width, device=None, pretrained=False):
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
    print('Loaded best state dict from epoch: {}'.format(checkpoint["epoch"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    return model


def load_model_from_checkpoint(storage_key, device="cpu"):
    model_weights_file = tempfile.NamedTemporaryFile()
    dl.Storage(auth=dl.Auth(client_id=config.ID,client_secret=config.SECRET)).get_file(storage_key, model_weights_file.name)
    model = Resnet(10, 64, device=device)
    model = load_checkpoint(model_weights_file.name, model, device)

    model_weights_file.close()
    print('Model loaded from DL Storage.')
    return model

    
if __name__ == "__main__":
    pass