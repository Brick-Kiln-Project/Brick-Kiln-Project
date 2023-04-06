"""
This is file from where the classification task of location brick kilns over india will be deployed. 
"""
from tasks import TaskGroup
from tqdm import tqdm
from tile import create_country_tiles
from utils import generate_unused_prefix
from itertools import repeat
import csv
import pickle as pkl
from descarteslabs.client.services.tasks import as_completed

import sys
sys.path.append("../")
import constants

def deploy_over_india():
    
    # first generate a prefix
    prefix = generate_unused_prefix()
    
    # then generate all of the tiles that model will be deployed over
    _, tile_keys = create_country_tiles()
    
    # create task group with proper prefixes
    tg = TaskGroup("model_deploy", f"{prefix}_model_deploy")
    
    # launch one task per tile
    for key in tile_keys:
        tg.deploy(key, constants.DEPLOY_MODEL_WEIGHTS, prefix)
        
def deploy_over_pakistan():
    prefix=generate_unused_prefix()
    _,tile_keys=create_country_tiles()
    tg=TaskGroup('model_deploy',f"{prefix}_model_deploy")
    rows=[["tile_key","geometry","prediction"]]
    with open('../Predicted_Datasets/'+prefix+'_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    #datarows=[]
    #for key in tile_keys:
    #    result=(tg.deploy(key,constants.DEPLOY_MODEL_WEIGHTS,prefix))
    #    datarows.append(result)
    datarows=tg.deploy(tile_keys,constants.DEPLOY_MODEL_WEIGHTS,prefix)
    count=0
    for x in as_completed(datarows,show_progress=False):
        print('Job #:',count,' of ',len(tile_keys))
        if x.is_success:
            try:
                data=x.result
            except:
                count+=1
                print('Failed!\nException:\nPickle error!')
                continue;
            with open('../Predicted_Datasets/'+prefix+'_results.csv', 'a', newline='') as file:
                print('Success!')
                print('\nSaving ',len(pkl.loads(data)),' items!')
                writer = csv.writer(file)
                writer.writerows(pkl.loads(data))
                print('Saved!')
        else:
            print('Failed!')
            print('\nException:\n',x.exception)
            print('\nLog:\n',x.log)
        count+=1

        
def test_deploy():
    """"""
    tg = TaskGroup("model_deploy", "test_deploy_single_tile")
    tg.deploy("2048:0:10.0:42:14:113", "kiln_prod_weights", "test_deploy_single_tile")

def test_storage():
    """"""
    tg = TaskGroup("storage_test", "storage_test")
    tg.deploy("storage_upload_test_file")
    
def test_model_loading():
    """ Runs successfully as of 4/22/22 """
    print("Testing loading model within Task.")
    tg = TaskGroup("model_loading_test", "model_loading_test")
    tg.deploy("kiln_prod_weights") # TODO: change this to use constants in production
    

def test_dataloader():
    print("Testing dataloader on single tile.")
    tg = TaskGroup("dataloader_test", "test_dataloader_on_single_tile")
    tg.deploy('2048:0:10.0:42:14:113')

def test_one_dummy_task():
    tg = TaskGroup("dummy", "test_one_dummy_task")
    tg.deploy()


def test_N_dummy_task(N=100):
    tg = TaskGroup("dummy", f"test_{N}_dummy_task")
    for i in tqdm(range(N)):
        tg.deploy()

def main(): 
    """
    Will contain 3 main functions:
        1) preprocess the AOI
        2) set up model args
        3) run the model via an aysnc function
    """

    #test_one_dummy_task()
    #test_N_dummy_task()
    #test_dataloader()
    #test_model_loading()
    #test_storage()
#     test_deploy()
    deploy_over_pakistan()


if __name__ == "__main__":
    main()
    print("Main finished running successfully.")
