import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Import Local Files
import sys

sys.path.append("../Configs/")
from keys import googleMapsKey
import help_texts
from global_func import GEELoadImage, masks2clouds

#Import relevant libraries!
from copy import deepcopy
from math import ceil
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input 
from keras.models import Model
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from kneed import KneeLocator
from random import shuffle
from tqdm import tqdm
from itertools import repeat
from io import BytesIO
from PIL import Image
from shapely.ops import transform
from googlemaps.maps import StaticMapPath

import pandas as pd
import pickle as pkl
import numpy as np
from pyproj import CRS, Transformer
import pathlib
import importlib.util
import shapely
import torch

#Import API's and Initialize them
import ee
ee.Initialize()
import googlemaps


def transformCoordinates(geometry,sourceCoord,toCoord):
    #Initialize Coord Swapping
    source = CRS(sourceCoord)
    to = CRS(toCoord)
    project = Transformer.from_crs(source, to, always_xy=True).transform
    
    #Create usable format for geometry and transform it
    shapegeom=shapely.geometry.shape(eval(geometry)['geometry'])
    transformedgeom = transform(project, shapegeom)
    
    #Get center of tile
    center=transformedgeom.centroid
    
    #Get coordinates of the transformed tile
    coordlist=[]
    xcoord,ycoord=transformedgeom.exterior.coords.xy
    for i in range(len(xcoord)):
        coordlist.append({"lat":ycoord[i],"lng":xcoord[i]})
        
    return center,coordlist


#Create extract_features function which takes the image and retrieves features from the vgg model
def extract_features(img,model,constants):
    reshaped_img=img.reshape(1,constants.GEE_IMAGE_SHAPE,constants.GEE_IMAGE_SHAPE,3)
    imgx=preprocess_input(reshaped_img)
    with torch.no_grad():
        features=model.predict(imgx,use_multiprocessing=False,verbose=0)
        return features


#Create gethrimage function which tries to obtain the satellite imagery of s2 and ab as uint8 arrays scaled to 0-255 for future plts, limited to TIMEOUT time in seconds.
def gethrimage(subtileGeometries):
    idx,geometry=subtileGeometries[0]
    lowResCollection,gmaps,constants=subtileGeometries[1]

    GEE_SATELLITE,GEE_START_DATE,GEE_END_DATE,GEE_FILTERS,GEE_FILTERS_BOUNDS,GEE_MAX_PIXEL_VALUE,GEE_IMAGE_FORMAT,GEE_IMAGE_SHAPE=constants
    """
    try:
        _ = ee.Geometry(eval(geometry)['geometry']).coveringGrid("EPSG:3857",700).getInfo()
    except:
        _ = ee.Geometry(eval(geometry)['geometry']).coveringGrid("EPSG:3857",700).getInfo()
    _=None
    """
    #Load google ee image
    try:
        collection=lowResCollection.filterBounds(ee.Geometry(eval(geometry)['geometry'])).map(masks2clouds)        

        image=collection.mean().clip(ee.Geometry(eval(geometry)['geometry']))
        buff=[GEE_MAX_PIXEL_VALUE,GEE_IMAGE_FORMAT,GEE_IMAGE_SHAPE]
        lowImg=GEELoadImage(image,eval(geometry)['geometry'],buff,rgb_only=True)
        if lowImg is None:
            print('None or bad image shape returned')
            return None
    except:
        print('Google Earth Error')
        return None
    
    #Load google maps static image
    center,coordlist=transformCoordinates(geometry,sourceCoord='EPSG:3857',toCoord='EPSG:4326')
    
    path=StaticMapPath(
        points=coordlist,
        weight=5,
        color="red",
        geodesic=True,
    )
    
    try:
        response = gmaps.static_map(
            size=(640,640),
            zoom=17,
            center=(center.y,center.x),
            maptype='satellite',
            format=GEE_IMAGE_FORMAT,
            scale='1',   
            path=path,
        )
        
        temp=b''.join(list(response))
        print(temp)
        with Image.open(BytesIO(temp)) as im:
            image=np.array(im.convert('RGB'))
        return [idx,lowImg,image,geometry,center]

    except Exception as e:
        print(e)
        return None


def InitiateModel(subtileGeometries,batch,root,constants):
    gmaps=googlemaps.Client(key=googleMapsKey)
    #Initialize image dictionaries with keys and None
    imageDict={}
    data={}
    for pair in subtileGeometries:
        imageDict[pair[0]]=None
    hrImgDict=deepcopy(imageDict)
    
    #Initialize satellites
    lowResCollection=ee.ImageCollection(constants.GEE_SATELLITE).filterDate(constants.GEE_START_DATE,constants.GEE_END_DATE)
    for filters,bounds in zip(constants.GEE_FILTERS,constants.GEE_FILTERS_BOUNDS):
        lowResCollection=lowResCollection.filter(ee.Filter.lte(filters,bounds))
        
    
    #Initialize model
    model = VGG16(include_top=False,input_shape=(constants.GEE_IMAGE_SHAPE,constants.GEE_IMAGE_SHAPE,3))
    model = Model(inputs = model.inputs, outputs = model.layers[-1].output)
    
    buff=[ constants.GEE_SATELLITE,constants.GEE_START_DATE,constants.GEE_END_DATE,constants.GEE_FILTERS,constants.GEE_FILTERS_BOUNDS,constants.GEE_MAX_PIXEL_VALUE,constants.GEE_IMAGE_FORMAT,constants.GEE_IMAGE_SHAPE]
    #Get all of the low res images using some threadpooling and extract features from the VGG model
    print('Getting Images!') 
    with Pool() as pool:
        with tqdm(total=len(subtileGeometries)) as pbar:
            for task in pool.imap_unordered(gethrimage, zip(subtileGeometries,repeat([lowResCollection,gmaps,buff],len(subtileGeometries)))):
                pbar.update()
                if task is not None and len(task)==5:
                    idx,lowImg,img,geometry,center=task
                    data[idx]=extract_features(lowImg,model,constants)
                    imageDict[idx]=[lowImg,geometry,center]
                    hrImgDict[idx]=[img,geometry,center]

    print('Done!')
    print(len(imageDict.keys()))
    if len(imageDict.keys()) <= 25:
        print('Not enough images processed, please check the api usages and errors to ensure enough data is being processed')
        return None
    
    #Upload the batch of images to the storage for later retrieval and clear the dictionaries
    print('Saving Image Dicts to Files!')
    loFile=open(constants.GROUPING_ROOT+root+'/'+root+'LrBatch'+str(batch),'wb')
    hrFile=open(constants.GROUPING_ROOT+root+'/'+root+'HrBatch'+str(batch),'wb')
    pkl.dump(imageDict,loFile)
    pkl.dump(hrImgDict,hrFile)
    loFile.close()
    hrFile.close()
    imageDict.clear()
    hrImgDict.clear()
            
    #initiate and run kmeans on a transformed feature set
    feat = np.array(list(data.values()))
    feat = feat.reshape(-1,2048)

    #Optimize the number of reduced features for groupings and run a final time on the most optimal
    print("Optimizing Features")
    pca = IncrementalPCA(n_components=20,batch_size=20)
    pca.fit(feat)
    varianceDiff = [x - pca.explained_variance_ratio_[i - 1] for i, x in enumerate(pca.explained_variance_ratio_)][1:]
    minVarianceIndex=varianceDiff.index(min(varianceDiff))+1

    pca = IncrementalPCA(n_components=minVarianceIndex,batch_size=20)
    pca.fit(feat)
    
    x=pca.transform(feat)
    feat=[]
    inertia=[]
    
    #Optimize Kmeans grouping by running it over a range of groups and run a final time on the most optimal
    print("Optimizing clusters")
    for y in range(1,constants.KMEANS_SIZE):
        kmeans=KMeans(n_clusters=y)
        kmeans.fit(x)
        inertia.append(kmeans.inertia_)
            
    kl = KneeLocator(range(1,constants.KMEANS_SIZE), inertia,direction='decreasing', curve="convex")
    inertia.clear()
    
    kmeans=KMeans(n_clusters=kl.knee)
    kmeans.fit(x)
    
    #Create Cluster Groups from KMeans results
    print("Creating groups!")
    groups={}
    for key, cluster in zip(np.array(list(data.keys())),kmeans.labels_):
        if str(cluster) not in groups.keys():
            groups[str(cluster)]=[]
            groups[str(cluster)].append(key)
        else:
            groups[str(cluster)].append(key)
    data.clear()
    
    #Upload the final grouping of the batch
    print("Uploading Batched Group to Storage!")
    groupFile=open(constants.GROUPING_ROOT+root+'/'+root+'GroupBatch'+str(batch),'wb')
    pkl.dump(groups,groupFile)
    groupFile.close()
    kmeans=None
    pca=None
    return None

def main(root,constants):
    if not os.path.exists(constants.GROUPING_ROOT+root):
        os.mkdir(constants.GROUPING_ROOT+root)
    totalDataset=pd.read_csv(constants.PREDICTION_ROOT+root+'_results.csv').to_dict()
    subtileGeometries=[]
    for keys in totalDataset['prediction']:
        subtileGeometries.append([totalDataset['idx'][keys],totalDataset['geometry'][keys]])
    shuffle(subtileGeometries)
    totalDataset.clear()
    
    print(len(subtileGeometries))
    batches = ceil(len(subtileGeometries)/constants.BATCH_SIZE)

    print("Face Tanking Initial Batch of Errors Please Ignore")
    InitiateModel(subtileGeometries[:12],0,root,constants)
    
    for batch in range(0,batches):
        print("Batch #",batch," of ", batches-1, "\n\n")
        if len(subtileGeometries)<constants.BATCH_SIZE+constants.KMEANS_SIZE:
            InitiateModel(subtileGeometries,batch,root,constants)
            del subtileGeometries
            return
        InitiateModel(subtileGeometries[:constants.BATCH_SIZE],batch,root,constants)
        del subtileGeometries[:constants.BATCH_SIZE]

def verify_dataset(datasetName,constants):
    if not os.path.exists(constants.PREDICTION_ROOT+datasetName+'_results.csv'):
        print('invalid->'+datasetName)
        return False
    else:
        return True
    
def initiate():
    argv=sys.argv[1:]
    constants=None
    if any(x in ['-co', '-config'] for x in argv):
        try:
            index=argv.index('-co')
        except:
            pass
        try: 
            index=argv.index('-config')
        except:
            pass
        file_path =str(pathlib.Path('../Configs/'+argv[index+1]))
        module_name = argv[index+1][:-3]
        del argv[index:index+2]
    else:
        file_path =str(pathlib.Path('../Configs/constants.py'))
        module_name = 'constants'
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)
    
    if len(argv)==1:
        if argv[0].lower() in ['-help', '-h']:
            print(help_texts.GROUPING_TEXT)
            return
    elif len(argv)%2 == 0:
        arg=[argv[i:i + 2] for i in range(0, len(argv), 2)]
        datasetName=constants.GROUPING_DATASET_NAME
        for instruction in arg:
            if instruction[0] in ['-dataset','-d']:
                datasetName=instruction[1]
            else:
                print('Bad keyword, '+instruction[0]+', expected [-dataset,-d]. Type -h or -help for help.')
                return
            
        if not verify_dataset(datasetName,constants):
            return
        
        main(datasetName,constants)
        
    else:
        print('Expected an even number of arguments (up to 2), got '+str(len(argv))+'. Type -h or -help for help.')
        
    
if __name__ == '__main__':
    initiate()