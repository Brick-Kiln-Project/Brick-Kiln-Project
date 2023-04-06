"""
File where each task type will be defined and ran.

2 steps to running a task: 
    1) 
"""
import sys
sys.path.append("../")
import constants
import descarteslabs as dl
sys.path.append("../UI_Labeling")
import config
from itertools import repeat

def model_deploy(*args):
    from model import load_model_from_checkpoint
    from time import time
    import descarteslabs as dl
    import json
    import numpy as np
    from time import time
    from dataloader import get_tile_DataLoader
    import torch
    import pickle as pkl
    
    tile_key, model_weights, file_prefix = args
    start_model_load = time()
    model = load_model_from_checkpoint(model_weights)
    model.eval()
    end_model_load = time()
    
    print(f"Took {end_model_load - start_model_load} secs to load model.")
    
    start_dataloader = time()
    loader = get_tile_DataLoader(tile_key)
    end_dataloader = time()
    
    print(f"Took {end_dataloader - start_dataloader} secs to load dataloader.")
    
    results_list = []
    start_classification = time()
    with torch.no_grad():
        for subtile_key, img, geometry in loader:
            if subtile_key[0]!='False':
                subtile_key = subtile_key[0]
                prediction = model(img)
                prediction = prediction.item()
                results_list.append([subtile_key,pkl.loads(geometry[0]),str(prediction)])
        
    end_classification = time()
    print(f"Took {end_classification - start_classification} secs to classify.")
    
    return pkl.dumps(results_list);
    #start_storage_upload = time()
    #dl.Storage(auth=dl.Auth(client_id=config.ID,client_secret=config.SECRET)).set(f"{file_prefix}_tile_results_{tile_key}", json.dumps(results_dict))
    #end_storage_upload = time()
    #print(f"Took {end_storage_upload - start_storage_upload} secs to upload.")
    
    
# need a task to test use of Storage object
def storage_test(*args):
    import descarteslabs as dl
    import json
    import numpy as np
    
    storage_file = args[0]
    keys = ["a", "b", "c", "d"] 
    values = ["1", "0", "1", "0"]
    d = {key : value for key, value in zip(keys, values)}
    dl.Storage(auth=dl.Auth(client_id=config.ID,client_secret=config.SECRET)).set(storage_file, json.dumps(d))


def model_loading_test(*args):
    """
    Tests loading in model weights.
    
    args[0] - DL Storage file containing trained model weights.
    """
    from model import load_model_from_checkpoint
    from time import time
    
    start = time()
    model = load_model_from_checkpoint(args[0])
    end = time()
    
    print(f"Model loaded successfully in {end - start} seconds.")
    
    
def dataloader_test(*args):
    """
    Expects one argument, the tile key.
    
    Runs successfully as a Task on 4/22/22. 
    """
    import time
    from dataloader import get_tile_DataLoader
    import torch
    import numpy as np
    
    tile_key = args[0]
    loader = get_tile_DataLoader(tile_key)
    total = []
    start = time.time()
    for subtile_key, img in loader:
        total.append(torch.mean(img))
    end = time.time()
    print(f"{end - start} seconds to iterate through.")
    print(np.mean(total))


def dummy_task_func(*args):
    """ Runs successfully as of 4/18/22. """
    print("Dummy task function successfully ran.")
    return 0


class TaskGroup():

    def __init__(self, task_func, task_name, maximum_concurrency=80, retry_count=1):
        # Select the function to create a task group around
        if task_func == "dummy":
            self.task_func = dummy_task_func
            self.requirements = None
            self.include_modules = None

        if task_func == "dataloader_test":
            self.task_func = dataloader_test
            self.requirements = None
            self.include_modules = ["constants", "dataloader"]
            
        if task_func == "model_loading_test":
            self.task_func = model_loading_test
            self.requirements = [
                "torch>=1.9.0",
                "torchvision>=0.10.0",
            ]
            self.include_modules = ["constants", "model"]
            
        if task_func == "model_deploy":
            self.task_func = model_deploy
            self.requirements = [
                "torch>=1.9.0",
                "torchvision>=0.10.0",
            ]
            self.include_modules = ["constants", "model", "dataloader", "utils","config"]
            
        if task_func == "storage_test":
            self.task_func = storage_test
            self.requirements = None
            self.include_modules = None

        # Set up task group
        tasks = dl.Tasks()
        self.async_func = tasks.create_function(
            f=self.task_func,
            name=task_name,
            image=constants.TASKS_DOCKER_IMAGE,
            maximum_concurrency=maximum_concurrency,
            include_modules=self.include_modules,
            requirements=self.requirements,
            retry_count=retry_count,
            task_timeout=2700,
            memory="72Gi",
        )



    def deploy(self, *args):
        task = self.async_func.map(args[0],repeat(args[1],len(args[0])),repeat(args[2],len(args[0])))
        return task




