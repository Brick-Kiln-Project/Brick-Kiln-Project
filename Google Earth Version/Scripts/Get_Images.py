import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import pathlib
import importlib.util
import numpy as np
import math
import pickle as pkl
import csv

import Get_Coordinates

sys.path.append("../Configs/")
"""
import constants
"""
import help_texts
import keys

from global_func import LoadUpsamplingModel
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import ee
service_account=keys.googleEarthAccount
credentials = ee.ServiceAccountCredentials(service_account,'../Configs/brick-kiln-project-d44b06c94881.json')
ee.Initialize(credentials)
import torch
import itertools

def create_CSV_from_dict_keys(confirmed,denied,lrimage,start,dataset,constants):
    columns=[]
    count=start
    
    #Iterate through our image dictionary and save the image array according to the respective classification
    print("Downloading Images to "+ constants.IMAGE_DATASETS_ROOT+dataset +" folders!")
    for key in tqdm(lrimage.keys()):
        folder=''
        if key in confirmed:
            label=1
            folder='positives/'
        elif key in denied:
            label=0
            folder='negatives/'
        else:
            continue;
        columns.append([folder+str(count)+'.npy',label,lrimage[key][1],key])
        np.save(constants.IMAGE_DATASETS_ROOT+dataset+'/'+folder+str(count),lrimage[key][0])
        count+=1
        
    #Check for existing metadata file and either add to or create using the collected column information    
    if os.path.exists(constants.IMAGE_DATASETS_ROOT+dataset+'/metadata.csv'):
        print("Adding to metadata.csv in "+constants.IMAGE_DATASETS_ROOT+dataset+"!")
        with open(constants.IMAGE_DATASETS_ROOT+dataset+'/metadata.csv','a',newline='') as file:
            writer = csv.writer(file)
            writer.writerows(columns)
    else:
        print("Writing to metadata.csv in "+constants.IMAGE_DATASETS_ROOT+dataset+"!")
        columns.insert(0,['Image','Label','Geometry','Key'])
        with open(constants.IMAGE_DATASETS_ROOT+dataset+'/metadata.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(columns)

            
def get_3_band_images(total_dt,batched,dataset,confirmed,denied,constants):
    #Create relevant folders
    keyImageDict={}
    if not os.path.exists(constants.IMAGE_DATASETS_ROOT+dataset):
        os.mkdir(constants.IMAGE_DATASETS_ROOT+dataset)
    if not os.path.exists(constants.IMAGE_DATASETS_ROOT+dataset+'/positives'):
        os.mkdir(constants.IMAGE_DATASETS_ROOT+dataset+'/positives')
    if not os.path.exists(constants.IMAGE_DATASETS_ROOT+dataset+'/negatives'):
        os.mkdir(constants.IMAGE_DATASETS_ROOT+dataset+'/negatives')
                
    #Chunk out the images to pass to getimage to save as we go            
    print('Getting Images!')
    steps=math.ceil(len(total_dt)/constants.DATASIZE)
    
    sr=LoadUpsamplingModel(constants)
    
    for x in range(0,steps):
        keyImageDict.clear()
        
        start=x*constants.DATASIZE
        if (x != steps-1):
            end=(x+1)*constants.DATASIZE
        else:
            end=(len(total_dt))
        batchedImages=[(batched[int(key)],key) for key in total_dt[start:end]]
        #Use threadpooling to run through each image
        print('Group#',x, 'of',steps-1)
        with tqdm(total=len(total_dt[start:end])) as pbar:
            for task in batchedImages:
                results=task
                img=results[0][0]
                result=sr.upsample(img)
                geom=results[0][1]
                key=results[1]
                keyImageDict[key]=(result,geom)
                pbar.update()
        
        #Pass the finished key-image dict to our saving function
        print('Saving this batch')
        create_CSV_from_dict_keys(confirmed,denied,keyImageDict,start,dataset,constants)
    print('Done!')


def main(dataset,constants):
    confirmedFile=open(constants.GROUPING_ROOT+dataset+'/'+dataset+'Confirmed','r+b')
    confirmed=pkl.load(confirmedFile)
    confirmedFile.close()

    deniedFile=open(constants.GROUPING_ROOT+dataset+'/'+dataset+'Denied','r+b')    
    denied=pkl.load(deniedFile)
    deniedFile.close()

    batched={}
    prefixed = [filename for filename in os.listdir(constants.GROUPING_ROOT+'/'+dataset) if filename.startswith(dataset+'LrBatch')]
    for file in prefixed:
        batchFile=open(constants.GROUPING_ROOT+dataset+'/'+file,'r+b')
        batch=pkl.load(batchFile)
        batched.update(batch)
        batchFile.close()

    total_dt={}
    total_dt.update(denied)
    total_dt.update(confirmed)
    get_3_band_images(list(total_dt),batched,dataset,confirmed,denied,constants)

def verify_folder(path,constants):
    if not os.path.exists(path):
        print('Invalid Path: ' +constants.GROUPING_ROOT+path)
        return False
    else:
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
            print(help_texts.IMAGES_TEXT)
            return
        elif args[0] in ['-coordinates','-cd']:
            dataset=constants.GENERATE_DATASET_NAME
            pathfolder=constants.GROUPING_ROOT+constants.GENERATE_DATASET_NAME
            
            if not verify_folder(pathfolder,constants):
                return
            main(dataset,constants)
            
            if not Get_Coordinates.verify_folder(constants.IMAGE_DATASETS_ROOT+dataset,constants):
                return
            if not Get_Coordinates.verify_prefix(constants.COORDINATE_POSTFIX):
                return
            if not Get_Coordinates.verify_model(constants.COORDINATE_MODEL,constants):
                return
            Get_Coordinates.main(dataset,constants.COORDINATE_POSTFIX,constants.COORDINATE_MODEL,constants)
            return
        else:
            print('1 argument found, expected [-coordinates,-c] instead got: '+args[0])
            return
        
    elif len(args)%2==0:
        dataset=constants.GENERATE_DATASET_NAME
        pathfolder=constants.GROUPING_ROOT+constants.GENERATE_DATASET_NAME
        arg=[args[i:i + 2] for i in range(0, len(args), 2)]
        for instruction in arg:
            if instruction[0] in ['-dataset','-d']:
                dataset=instruction[1]
                pathfolder=constants.GROUPING_ROOT+instruction[1]
            else:
                print('Bad keyword, '+instruction[0]+', expected [-dataset,-d].')
                return
        if not verify_folder(pathfolder,constants):
            return
        main(dataset,constants)
    else:
        print('Expected even number of arguments (up to 2) found: '+str(len(args)))
    
if __name__=="__main__":
    initiate()
    