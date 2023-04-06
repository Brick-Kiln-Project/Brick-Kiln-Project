import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#Import Local Files
import sys
"""
sys.path.append("../Configs/")
import constants
"""
#Import Libraries
import pathlib
import importlib.util
import csv
import pyproj
import geojson
import pandas as pd
import geopandas as gpd
import numpy as np

from shapely import geometry
from random import sample
from tqdm import tqdm
from shapely.ops import transform


def main(datasetName,country,baseDataset,baseCountry,constants):
    gdfp,highDataset,restDataset = CreateDatasetDataFrame(baseDataset,constants)
    max_area,gdf,shapelyStates = CreateStateDataFrame(baseCountry,constants)
    ans = gpd.tools.sjoin(gdfp, gdf, predicate="within", how='left')
    ans.drop('geometry',axis=1).drop('index_right',axis=1).to_csv(constants.STATE_SAMPLING_ROOT+country)
    SampleDatasets(datasetName,shapelyStates,highDataset,restDataset,max_area,ans,constants)
    return datasetName

        
def CreateStateDataFrame(countryName,constants):
    #Create shapely brick belt
    with open(constants.BRICK_BELT) as f:
        brick_belt = geojson.load(f) 
    with open(constants.GEOJSON_ROOT+countryName+'_States.geojson') as f:
        states=geojson.load(f)
    shapelyBelt=geometry.shape(brick_belt['features'][0]['geometry'])
    
    #Prepare to make a geodataframe from the geometry of the country states, also take the time to calculate non-belt max area
    shapelyStates={}
    togpd=[]
    names=[]
    within={}
    max_area=0
    for x in states['features']:
        
        if shapelyBelt.intersects(geometry.shape(x['geometry'])):
            within = shapelyBelt.intersection(geometry.shape(x['geometry'])).area/geometry.shape(x['geometry']).area >= .25
        else:
            within=False
        
        name_var='NAME_1'
        
        if 'name' in x['properties'].keys():
            name_var='name'
        
        shapelyStates[x['properties'][name_var]]={
            'geometry':geometry.shape(x['geometry']),
            'within':within,
            'tiles':[]
        }
        if not shapelyStates[x['properties'][name_var]]['within']:
            if max_area<shapelyStates[x['properties'][name_var]]['geometry'].area:
                max_area=shapelyStates[x['properties'][name_var]]['geometry'].area
        togpd.append(geometry.shape(x['geometry']))
        names.append(x['properties'][name_var])
        
    #Create state geodataframe
    gdf = gpd.GeoDataFrame(data={'name':names}, crs='epsg:4326', geometry=togpd)
    return max_area,gdf,shapelyStates
        
        
def SampleDatasets(datasetFile,shapelyStates,highDataset,restDataset,max_area,ans,constants):
    with open(constants.PREDICTION_ROOT+datasetFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([['idx','geometry','prediction','area']])
    
    stateTiles={}
    for x in shapelyStates.keys():
        stateTiles[x]=[]
    for x in tqdm(range(len(ans['idx']))):
        if type(ans['name'][x])==str:
            stateTiles[ans['name'][x]].append([ans['geom'][x],ans['idx'][x]])
                   
    #Iterate through each state and sample respective amounts per state, then save it to the sampled csv.    
    for x in shapelyStates.keys():
        print('Working on '+x+'!')
        
        if x in constants.OVERSAMPLE_STATES:
            print('Oversampling!')
            modifier=constants.OVERSAMPLE_MODIFIER
        else:
            modifier=1
                   
        if shapelyStates[x]['within']:
            print('Inside Brick Belt!')
            samplesize=min([len(stateTiles[x]),round(constants.MAX_KILN_STATE_SAMPLE*modifier)])
        else:
            print('Outside Brick Belt!')
            cur_area=shapelyStates[x]['geometry'].area
            samplesize=min([round(constants.MAX_NON_KILN_STATE_SAMPLE*cur_area/max_area*modifier),len(stateTiles[x])])
            if round(constants.MIN_NON_KILN_STATE_SAMPLE*modifier) > len(stateTiles[x]):
                samplesize=len(stateTiles[x])
            else:
                samplesize=max([round(constants.MIN_NON_KILN_STATE_SAMPLE*modifier),samplesize])
        print('Sampling '+str(samplesize)+' tiles!')
        shapelyStates[x]['tiles']=sample(stateTiles[x],samplesize)
        rows=[]
        print('Saving!')
        for y in range(len(shapelyStates[x]['tiles'])):
            rows.append([highDataset['idx'][shapelyStates[x]['tiles'][y][1]],highDataset['geometry'][shapelyStates[x]['tiles'][y][1]],highDataset['prediction'][shapelyStates[x]['tiles'][y][1]],x])
        with open(constants.PREDICTION_ROOT+datasetFile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    if constants.RANDOM_SAMPLE:
        print('Sampling randomly!')
        sampledkeys=sample(list(restDataset['geometry'].keys()),min([len(restDataset['geometry'].keys()),constants.RANDOM_SAMPLE]))
        print('Sampling',len(sampledkeys),'tiles!')
        rows=[]
        print('Saving!')
        for x in sampledkeys:
            rows.append([restDataset['idx'][x],restDataset['geometry'][x],restDataset['prediction'][x],'random'])
        with open(constants.PREDICTION_ROOT+datasetFile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    
def CreateDatasetDataFrame(dataset,constants):
    #Create Datasets
    highDataset={'idx':{},'geometry':{},'prediction':{},'area':{}}
    restDataset={'idx':{},'geometry':{},'prediction':{},'area':{}}
    totalDataset=pd.read_csv(constants.PREDICTION_ROOT+dataset+"_results.csv")
    highcount=0
    lowcount=0
    for idx in tqdm(totalDataset.index):
        if totalDataset['prediction'][idx] >= constants.PREDICTION_CONFIDENCE_BOUNDS:
            highDataset['idx'][highcount]=totalDataset['idx'][idx]
            highDataset['geometry'][highcount]=totalDataset['geometry'][idx]
            highDataset['prediction'][highcount]=totalDataset['prediction'][idx]
            highcount+=1
        else:
            restDataset['idx'][lowcount]=totalDataset['idx'][idx]
            restDataset['geometry'][lowcount]=totalDataset['geometry'][idx]
            restDataset['prediction'][lowcount]=totalDataset['prediction'][idx]
            lowcount+=1
    #Iterate through all of our keys and obtain the geometric center and the respective tile_keys
    centers=[]
    tile_idxs=[]
    geoms=[]

    source = pyproj.CRS('EPSG:3857')
    to = pyproj.CRS('EPSG:4326')
    project = pyproj.Transformer.from_crs(source, to, always_xy=True).transform

    for x in tqdm(highDataset['geometry'].keys()):
        shapegeom=geometry.shape(eval(highDataset['geometry'][x])['geometry'])
        transformedgeom = transform(project, shapegeom)
        centers.append(transformedgeom.centroid)
        tile_idxs.append(highDataset['idx'][x])
        geoms.append(shapegeom)

    #Create the tile geodataframe from the center points 
    gdfp = gpd.GeoDataFrame(data={'img_idx':tile_idxs,'idx':range(len(tile_idxs)),'geom':geoms},crs='epsg:4326', geometry=centers)
    
    return gdfp,highDataset,restDataset

def verify_country(country,dataset_name,prefix,constants):
    if not os.path.exists(constants.GEOJSON_ROOT+country+'_States.geojson'):
        print('Geojson State file not found, please try again: '+constants.GEOJSON_ROOT+country+'_States.geojson')
        return False
    elif os.path.exists(constants.STATE_SAMPLING_ROOT+prefix+dataset_name+'_'+country+'_results.csv'):
        cont=True
        while(cont):
            inp=input(constants.STATE_SAMPLING_ROOT+prefix+dataset_name+'_'+country+'_results.csv'+' already exists, would you like to overwrite? (y/n):')
            if inp=='y':
                return True
            elif inp=='n':
                print('Aborting, please change your arguments.')
                return False
            else:
                print('Not a valid response, try again')
    else:    
        return True
    
def verify_dataset(dataset_name,prefix,constants):
    if not os.path.exists(constants.PREDICTION_ROOT+dataset_name+'_results.csv'):
        print('Provided Dataset_Name results not found, please try again: '+constants.PREDICTION_ROOT+dataset_name+'_results.csv')
        return False
    elif os.path.exists(constants.PREDICTION_ROOT+prefix+dataset_name):
        cont=True
        while(cont):
            inp=input(constants.PREDICTION_ROOT+prefix+dataset_name+' already exists, would you like to overwrite? (y/n):')
            if inp=='y':
                return True
            elif inp=='n':
                print('Aborting, please change your prefix or dataset.')
                return False
            else:
                print('Not a valid response, try again')
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
        if args[0] in ['-help','-h']:
            print(help_texts.SAMPLING_TEXT)
            return
    elif len(args)%2 == 0:
        country=constants.SAMPLING_COUNTRY
        prefix=constants.SAMPLING_DATASET_PREFIX
        dataset=constants.SAMPLING_DATASET_NAME
        
        arg=[args[i:i + 2] for i in range(0, len(args), 2)]
        for instruction in arg:
            if instruction[0].lower() in ['-country','-c']:
                country=instruction[1]
            elif instruction[0].lower() in ['-prefix','-p']:
                prefix=instruction[1]
            elif instruction[0].lower() in ['-dataset','-d']:
                dataset=instruction[1]
            else:
                print('Bad keyword, '+instruction[0]+', expected [-prefix,-p], [-country,-c] or [-dataset,-d].')
                return
        
        #Verify Prefix is valid
        if not verify_prefix(prefix):
            return
        
        #Verify dataset path exists
        if not verify_dataset(dataset,prefix,constants):
            return
        
        #Verify country exists in the model
        if not verify_country(country,dataset,prefix,constants):
            return
        
        
        main(prefix+dataset+'_results.csv',prefix+dataset+'_'+country+'_results.csv',dataset,country,constants)
        
    else:
        print('Expected an even number of arguments (up to 6), got '+str(len(args)))
        return
    
    print("Done.")
    
    
if __name__ == '__main__':
    initiate()
    