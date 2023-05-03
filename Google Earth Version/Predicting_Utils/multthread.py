import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from model import load_model_from_checkpoint
from dataloader import get_tile_DataLoader
from tqdm import tqdm
from itertools import repeat
import torch
import numpy as np
import pickle as pkl
import csv
from time import sleep
import json
from torch.autograd import Variable
from torch.nn import functional as F

import sys
sys.path.append("../Configs/")
#import constants
from global_func import LoadUpsamplingModel
sys.path.append("../Scripts/")
from Get_Coordinates import returnCAM, UpsampledResnet, load_checkpoint

def evaluate(geometry):
    #Create Dataloader from tile_geometry
    idx_mult,geom=geometry[0]
    country,constants=geometry[1]
    dataloader=get_tile_DataLoader(geom,idx_mult,country,constants,rgb_only=True)
    return dataloader

def coordinateFunc(model,image,idx,constants,geom):
    latlons={}
    upsamplednet = UpsampledResnet(3,64, device=constants.CUDA)
    upsamplednet = load_checkpoint(constants.MODEL_ROOT+model+"_50_training_steps/checkpoints/best_dl_best.pth", 
                                     upsamplednet, 
                                     'cpu')
    upsamplednet.to(constants.CUDA)
    upsampled_conv_name='layer4'
    upsamplednet.eval()

    upsampled_features_blobs = []
    def hook_feature(module, input, output):
        upsampled_features_blobs.append(output.data.cpu().numpy())
    upsamplednet._modules.get('resnet')._modules.get(upsampled_conv_name).register_forward_hook(hook_feature)

    upsampledparams = list(upsamplednet.parameters())
    upsampled_weight_softmax = [np.dot(np.squeeze(upsampledparams[-2].data.cpu().numpy()),(np.squeeze(upsampledparams[-4].data.cpu().numpy())))]
    
    upsampledimg=image.float().to(constants.CUDA)
    """
    upsampledimg=np.transpose(upsampledimg,(2,0,1))
    upsampledimg=torch.Tensor(upsampledimg).to(constants.CUDA)
    """
    upsampledlogit=upsamplednet(Variable(upsampledimg))
    upsampledh_x=F.softmax(upsampledlogit,dim=1).data.squeeze()
    upsampledprobs,upsampledidx=upsampledh_x.sort(0,True)
    upsampledidx=upsampledidx.cpu().numpy()
    upsampledCAMs,upsampledlabeled,upsamplednr_objects,upsampledcenters=returnCAM(upsampled_features_blobs[-1],[upsampled_weight_softmax[-1]*-1],[upsampledidx],9,upsampledimg.shape[1])
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


    latlons[str(idx[0])]=temp
    upsampled_features_blobs=[]
    return latlons

def startthread(geometries,country,prefix,model_path,constants,coordfile):
    if coordfile:
        with open (constants.COORDINATES_ROOT+coordfile+'.json','w') as outfile:
            json.dump({},outfile)
        
    numbered_geometries=list(enumerate(geometries))
    with torch.no_grad():
        #Initialize upsampling and standard model
        model=load_model_from_checkpoint(model_path,constants,constants.CUDA)
        model.eval()
        sr=LoadUpsamplingModel(constants)
        buff=[ constants.GEE_SATELLITE,constants.GEE_START_DATE,constants.GEE_END_DATE,constants.GEE_FILTERS,constants.GEE_FILTERS_BOUNDS,constants.GEE_MAX_PIXEL_VALUE,constants.GEE_IMAGE_FORMAT,constants.GEE_IMAGE_SHAPE]
        #Iterate through the tile geometries and create a dataloader to predict upon 
        with tqdm(total=len(numbered_geometries)) as pbar:
            with ThreadPool(processes=10) as pool:
                for task in pool.imap(evaluate,zip(numbered_geometries,repeat([country,buff],len(numbered_geometries)))):
                    pbar.update()
                    try:
                        dataloader=task
                    except:
                        print('Unknown Complete Tile Error')
                        continue;
                    #print('Classifying')
                    results_list=[]
                    latlons={}
                    for subtile_geometry,img,idx in tqdm(dataloader):
                        if subtile_geometry!='False':
                            subtile_geo=pkl.loads(subtile_geometry)
                            result=sr.upsample(img.numpy().squeeze())
                            img=torch.Tensor(np.expand_dims(np.transpose(result,(2,0,1)),axis=0))
                            prediction=model(img.float().to(constants.CUDA)).to(constants.CUDA)
                            prediction=prediction.item()
                            results_list.append([int(idx[0]),subtile_geo,str(prediction)])
                            
                            if coordfile and (prediction > .9):
                                latlons.update(coordinateFunc(model_path,img,idx,constants,subtile_geo))
                    if coordfile:
                        temp = {}
                        with open (constants.COORDINATES_ROOT+coordfile+'.json','r+') as outfile:
                            temp=json.load(outfile)
                            temp.update(latlons),outfile
                        with open (constants.COORDINATES_ROOT+coordfile+'.json','w+') as outfile:
                            json.dump(temp,outfile)
                        
                    with open(constants.PREDICTION_ROOT+prefix+'_results.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerows(results_list)
                    
                    results_list.clear()
                    dataloader=None
    return None