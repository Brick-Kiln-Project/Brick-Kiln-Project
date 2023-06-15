"""
This is file from where the classification task of location brick kilns over a country will be deployed. 
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Local Files and Utils
import sys

import Group_Predictions
import Sample_Predictions

sys.path.append('../Predicting_Utils/')
from tile import create_country_tiles
from utils import generate_unused_prefix
from multthread import startthread

sys.path.append("../Configs/")
#import constants
import help_texts
import keys

#Necessary Libraries
import pathlib
import importlib.util
import os
import csv
import ee
import numpy as np
#Initialize ee
service_account=keys.googleEarthAccount
credentials = ee.ServiceAccountCredentials(service_account,'../Configs/brick-kiln-project-d44b06c94881.json')
ee.Initialize(credentials)

            
def deploy_over_country(country,model,constants,coordfile):
    #Generate tile geometries
    print('Tiling country')
    tile_geometries,country=create_country_tiles(constants,country,constants.GEE_COUNTRY_DATASET)
    
    #Ensure data is valid
    if tile_geometries is None:
        return None
    
    print('Got',int(len(tile_geometries)))
    
    #Generate unused prefix for dataset
    print('Getting Prefix')
    prefix=generate_unused_prefix(constants)
    print('Got',prefix)
    
    #Initiate Result csv
    print('Initiating '+prefix+'_results.csv')
    rows=[["idx","geometry","prediction"]]
    with open(constants.PREDICTION_ROOT+prefix+'_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    
    print('Start Threading')
    startthread(tile_geometries,country,prefix,model,constants,coordfile)
    return prefix

def main(country,model,constants,coordfile): 
    return deploy_over_country(country,model,constants,coordfile)

def verify_country(country,constants):
    location = ee.FeatureCollection(constants.GEE_COUNTRY_DATASET).filter(ee.Filter.eq('country_na',country))
    if not location.geometry().getInfo()['coordinates']:
        print('Invalid country, please try again: '+country)
        return False
    else:
        return True
        
def verify_model(model,constants):
    if not os.path.exists(constants.MODEL_ROOT+model+'_50_training_steps/checkpoints/best_dl_best.pth'):
        print('Invalid model path, please try again: '+constants.MODEL_ROOT+model+'_50_training_steps/checkpoints/best_dl_best.pth does not exist.')
        return False
    else:
        return True

def verify_coordfile(coordfile,constants):
    if (type(coordfile) == type(False)) and not coordfile:
        return True
    for char in coordfile:
        if not char.isalnum() and char not in '-_':
            print('Provided coordinate file contains a character that is not alphanumeric or one of "-_", please try again: '+coordfile)
            return False
    if os.path.exists(constants.COORDINATES_ROOT+coordfile+'.json'):
        print('Provided coordinate file already exists, please delete the original or change your coordinate file name')
        return False
    
    return True
    
def initiate():
    #Check arguments
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
        if args[0].lower() in ['-help', '-h']:
            print(help_texts.PREDICTION_TEXT)
            return
        elif args[0].lower() in ['-full','-f']:
            cont=True
            while(cont):
                inp=input('Running full suite using variables in '+module_name+', confirm? [y,n]')
                if inp=='y':
                    cont=False
                elif inp=='n':
                    print('Aborting, please change constants.py to fit your needs!')
                    return
                else:
                    print('Not a valid response, try again')
                    
            #Verify country exists in the model
            if not verify_country(constants.DEPLOY_COUNTRY,constants):
                return
            
            #Verify model path exists
            if not verify_model(constants.DEPLOY_MODEL_WEIGHTS,constants):
                return
            
            if not Sample_Predictions.verify_prefix(constants.SAMPLING_DATASET_PREFIX):
                return
            
            prefix=main(constants.DEPLOY_COUNTRY,constants.DEPLOY_MODEL_WEIGHTS,constants,False)
            print('Finished Predicting')
            
            if not Sample_Predictions.verify_dataset(prefix,constants.SAMPLING_DATASET_PREFIX,constants):
                return
            
            if not Sample_Predictions.verify_country(constants.SAMPLING_COUNTRY,prefix,constants.SAMPLING_DATASET_PREFIX,constants):
                return
            
            root=Sample_Predictions.main(constants.SAMPLING_DATASET_PREFIX+prefix+'_results.csv',constants.SAMPLING_DATASET_PREFIX+prefix+'_'+constants.SAMPLING_COUNTRY+'_results.csv',prefix,constants.SAMPLING_COUNTRY,constants)
            print('Finished Sampling')
            
            if not Group_Predictions.verify_dataset(constants.SAMPLING_DATASET_PREFIX+prefix,constants):
                return
            print('Finished Verifying')
            Group_Predictions.main(constants.SAMPLING_DATASET_PREFIX+prefix,constants)
            print('Finished Grouping')
            
        else:
            print('Got a single argument and expected [-full,-f] but got, '+args[0]+ '. Type -h or -help for help.')
            
    elif len(args)%2 == 0:
        country=constants.DEPLOY_COUNTRY
        model=constants.DEPLOY_MODEL_WEIGHTS
        coordfile=False
        
        arg=[args[i:i + 2] for i in range(0, len(args), 2)]
        for instruction in arg:
            if instruction[0].lower() in ['-country','-c']:
                country=instruction[1]
            elif instruction[0].lower() in ['-model','-m']:
                model=instruction[1]
            elif instruction[0].lower() in ['-coordinates','-coor']:
                coordfile=instruction[1]
            else:
                print('Bad keyword, '+arg[0]+', expected [-full,-f], [-country,-c] or [-model,-m]. Type -h or -help for help.')
                return
            
        #Verify country exists in the model
        if not verify_country(country,constants):
            return
        #Verify model path exists
        if not verify_model(model,constants):
            return
        #Verify coordfile is valid and not already existing
        if not verify_coordfile(coordfile,constants):
            return
        
        main(country,model,constants,coordfile)
        
    else:
        print('Expected a single argument [-full,-f] or an even number of arguments (up to 4), got '+str(len(args))+'. Type -h or -help for help.')

    
    print("Done.")

if __name__ == "__main__":
    initiate()