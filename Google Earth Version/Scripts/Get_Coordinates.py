# Silence Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Import Files
import sys

sys.path.append("../Configs/")
import help_texts
"""
import constants
"""
#Import Libraries
import torch
import cv2
import json
import pathlib
import importlib.util
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import functional as F
from scipy import ndimage
from torchvision.models import resnet18


class UpsampledResnet(nn.Module):
    def __init__(self, num_channels, image_width, device=None, pretrained=False):
        super(UpsampledResnet, self).__init__()
        self.device = device        
        self.resnet = resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(num_channels, image_width, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.upsample = nn.Upsample(scale_factor=8)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 256).to(device)
        self.final_fc = nn.Linear(256, 1).to(device)
        
        self.resnet = self.resnet.to(device)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x=self.upsample(x)
        resnet_output = self.resnet(x)
        outputs = self.final_fc(resnet_output)
        outputs = self.sigmoid(outputs)
        return outputs
    
def load_checkpoint(model_checkpoint, model, device, optimizer=None):
    """
    Loads a pretrained checkpoint to continue training
    model_checkpoint: Path of the model_checkpoint that ends with .pth
    model: model to load to
    device: devide to load on (gpu/cpu)
    optimizer (optional): optimize to load state to
    """
    checkpoint = torch.load(model_checkpoint, map_location=device)
    #print('Loaded best state dict from epoch: {}'.format(checkpoint["epoch"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    return model

def returnCAM(feature_conv, weight_softmax, class_idx, kernel, size_upsample):
    # generate the class activation maps upsample to 256x256
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    labeled=[]
    nr_objects=[]
    centers=[]
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = np.transpose(cam)
        #normalize
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        #calculate mean and convolute using large mask
        #"""
        mean = np.mean(cam[np.nonzero(cam)])
        cam = cam-mean
        cam = np.clip(cam,0,None)
        cam = cv2.GaussianBlur(cam,(kernel,kernel),0)
        
        
        #calculate mean and convolute using medium mask
        #"""
        mean = np.mean(cam[np.nonzero(cam)])
        cam = cam-mean
        cam = np.clip(cam,0,None)
        
        cam = cv2.GaussianBlur(cam,(kernel,kernel),0)
        #"""
        
        #calculate mean and convolute using small mask
        #"""
        mean = np.mean(cam[np.nonzero(cam)])
        cam = cam-mean
        cam = np.clip(cam,0,None)
        #"""
        
        #calculate mean and send the final cam
        #"""
        mean = np.mean(cam[np.nonzero(cam)])
        cam = cam-mean
        #"""
        
        cam=cv2.resize(cam, (size_upsample,size_upsample))
        cam_img = np.clip(cam,0,None)
        cam_img = cam_img-(np.max(cam)*.75)
        cam_img=np.clip(cam,0,None)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
        labeled1,nr_objects1=ndimage.label(output_cam)
        labeled.append(labeled1)
        nr_objects.append(nr_objects1)
        centers1=ndimage.center_of_mass(output_cam[-1].squeeze(),labeled[-1].squeeze(),range(1,nr_objects1+1))
        centers.append(centers1)
    return output_cam,labeled,nr_objects,centers

def returnOriginalCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam=np.transpose(cam)
        
        #normalize
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        cam_img = np.clip(cam,0,None)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cam_img)
        
    return output_cam

def main(dataset,postfix,model,constants):
    df = pd.read_csv(constants.IMAGE_DATASETS_ROOT+dataset+"/metadata.csv")
    
    upsamplednet = UpsampledResnet(3,64, device='cuda')
    upsamplednet = load_checkpoint(constants.MODEL_ROOT+model+"_50_training_steps/checkpoints/best_dl_best.pth", 
                                     upsamplednet, 
                                     'cpu')
    upsamplednet.to('cuda')
    upsampled_conv_name='layer4'
    upsamplednet.eval()

    upsampled_features_blobs = []
    def hook_feature(module, input, output):
        upsampled_features_blobs.append(output.data.cpu().numpy())
    upsamplednet._modules.get('resnet')._modules.get(upsampled_conv_name).register_forward_hook(hook_feature)

    upsampledparams = list(upsamplednet.parameters())
    upsampled_weight_softmax = [np.dot(np.squeeze(upsampledparams[-2].data.cpu().numpy()),(np.squeeze(upsampledparams[-4].data.cpu().numpy())))]

    with tqdm(total=len(df['Geometry'])) as pbar:
        latlons={}
        for idx,x in enumerate(df['Geometry']):
            if df['Label'][idx]:
                geom=eval(df['Geometry'][idx])
                image=df['Image'][idx]
                upsampledimg=np.load(constants.IMAGE_DATASETS_ROOT+dataset+'/'+image)
                upsampledimg=np.transpose(upsampledimg,(2,0,1))
                upsampledimg=torch.Tensor(upsampledimg).to('cuda')
                upsampledlogit=upsamplednet(Variable(upsampledimg.unsqueeze(0)))
                upsampledh_x=F.softmax(upsampledlogit,dim=1).data.squeeze()
                upsampledprobs,upsampledidx=upsampledh_x.sort(0,True)
                upsampledprobs=upsampledprobs.cpu().numpy()
                upsampledidx=upsampledidx.cpu().numpy()
                upsampledCAMs,upsampledlabeled,upsamplednr_objects,upsampledcenters=returnCAM(upsampled_features_blobs[-1],[upsampled_weight_softmax[-1]*-1],[upsampledidx],9,upsampledimg.shape[1])
                upsampledoCAMs = returnOriginalCAM(upsampled_features_blobs[-1],[upsampled_weight_softmax[-1]*-1],[upsampledidx])
                temp=[]
                for centers in upsampledcenters[0]:
                    x1=geom['geometry']['coordinates'][0][0][0]
                    y1=geom['geometry']['coordinates'][0][0][1]
                    x2=geom['geometry']['coordinates'][0][3][0]
                    y2=geom['geometry']['coordinates'][0][3][1]
                    xstep=abs(x1-x2)/256*centers[1]
                    ystep=abs(y1-y2)/256*centers[0]
                    lat=round(ystep+min(y1,y2))
                    lon=round(xstep+min(x1,x2))
                    temp.append((lat,lon))


                latlons[str(df['Key'][idx])]=temp
                upsampled_features_blobs=[]


            pbar.update()

        with open (constants.COORDINATES_ROOT+dataset+postfix+'.json','w') as outfile:
            json.dump(latlons,outfile)
            
def verify_folder(path,constants):
    if not os.path.exists(path):
        print('Invalid Path: ' +constants.GROUPING_ROOT+path)
        return False
    else:
        return True
    
def verify_model(model,constants):
    if not os.path.exists(constants.MODEL_ROOT+model+'_50_training_steps/checkpoints/best_dl_best.pth'):
        print('Invalid model path, please try again: '+constants.MODEL_ROOT+model+'_50_training_steps/checkpoints/best_dl_best.pth')
        return False
    else:
        return True
    
def verify_prefix(prefix):
    for char in prefix:
        if not char.isalnum() and char not in '-_ ':
            print('Provided Prefix contains a character that is not alphanumeric or one of "-_ ", please try again: '+prefix)
            return False
    return True

def initiate():
    args=sys.argv[1:]
    constants=None
    if any(x in ['-co', '-config'] for x in args):
        try:
            index=args.index('-co')
        except:
            pass
        try: 
            index=args.index('-config')
        except:
            pass
        file_path =str(pathlib.Path('../Configs/'+args[index+1]))
        module_name = args[index+1][:-3]
        del args[index:index+2]
    else:
        file_path =str(pathlib.Path('../Configs/constants.py'))
        module_name = 'constants'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    
    if len(args)==1:
        if args[0] in ['-h','-help']:
            print(help_texts.COORDINATE_TEXT)
            return
    elif len(args)%2==0:
        dataset=constants.COORDINATE_DATASET_NAME
        postfix=constants.COORDINATE_POSTFIX
        model=constants.COORDINATE_MODEL
        pathfolder=constants.IMAGE_DATASETS_ROOT+dataset
        arg=[args[i:i + 2] for i in range(0, len(args), 2)]
        for instruction in arg:
            if instruction[0] in ['-dataset','-d']:
                dataset=instruction[1]
                pathfolder=constants.IMAGE_DATASETS_ROOT+instruction[1]
            elif instruction[0] in ['-postfix','-p']:
                postfix=instruction[1]
            elif instruction[0] in ['-model','-m']:
                model=instruction[1]
            else:
                print('Bad keyword, '+instruction[0]+', expected [-dataset,-d],[-postfix,-p], or [-model,-m].')
                return
        if not verify_folder(pathfolder,constants):
            return
        if not verify_model(model,constants):
            return
        if not verify_prefix(postfix):
            return
        main(dataset,postfix,model,constants)
    else:
        print('Expected an even number of arguments (up to 6) found: '+str(len(args)))

if __name__ == '__main__':
    initiate()