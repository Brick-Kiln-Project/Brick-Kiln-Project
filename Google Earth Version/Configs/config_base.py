import pathlib

###COMMON VARIABLES
#Predict_Country.py
DEPLOY_MODEL_WEIGHTS = str() #Name of the model located in 'Google Earth Version/Data/Model Logs/'
DEPLOY_COUNTRY = str() #Country to tile, should be in the GEE COUNTRY DATASET collection

#Sample_Predictions.py
SAMPLING_DATASET_NAME = str() #Existing Dataset in 'Google Earth Version/Data/Predicted Datasets/'. Relevant when running Sample_Predictions.py alone
SAMPLING_COUNTRY = str() #Country of SAMPLING_DATASET_NAME, should have an accompanying GEOJSON file in 'Google Earth Version/GeoJSONS/'. Relevant when running Sample_Predictions.py alone
SAMPLING_DATASET_PREFIX = str() #String to call the sampled dataset

#Group_Predictions.py
GROUPING_DATASET_NAME = str() #Existing Sampled Dataset in 'Google Earth Version/Data/Predicted Datasets/'. Relevant when running Group_Predictions.py alone

#Get_Imaegs.py
GENERATE_DATASET_NAME = str() #Existing Sampled Dataset in 'Google Earth Version/Data/Predicted Datasets/'.

#Get_Coordinates.py
COORDINATE_MODEL = str() #Name of the model located in 'Google Earth Version/Data/Model Logs/'
COORDINATE_DATASET_NAME = str() #Image Datasets, Dataset Name
COORDINATE_POSTFIX = str() #String to call the coordinate file

#Train_Model.py
TRAINING_MODEL_NAME = str() #Name of the model to save 
TRAINING_DATASET_LIST=[ 
    str(),
] #Datasets found in 'Google Earth Version/Data/Image Datasets' to train/evaluate model on


###ADVANCED VARIABLES
#Predict_Country.py
DEPLOY_START_FROM=0 #Initialize Script from x tile in case of early termination or stuck on loop

#Sample_Predictions.py
MAX_KILN_STATE_SAMPLE=500 #Maximum number of tiles to sample from states within the brick belt geojson
MAX_NON_KILN_STATE_SAMPLE=250 #Maximum number of tiles to sample from the largest state not inside the brick belt geojson
MIN_NON_KILN_STATE_SAMPLE=50 #Maximum number of tiles to sample from the smallest state not inside the brick belt geojson, the rest of the states are sampled uniformly to scale with their size in comparison
RANDOM_SAMPLE=500 #Number of randomly sampled tiles
OVERSAMPLE_STATES=[
    'F.A.T.A',
    'N.W.F.P',
    'Punjab'
] #States to oversample from
OVERSAMPLE_MODIFIER=1.5 #Ratio at which to oversample the oversampled states 
PREDICTION_CONFIDENCE_BOUNDS=.90 #Confidence Bounds within which to accept tiles for sampling

#Group_Predictions.py
BATCH_SIZE=1100 #Number of images to acquire at a time and save into pkl dump files (MEMORY LIMITED, likely suffers from memory leaks, adjust as necessary) (BEWARE, calls to this function will increase google static maps calls which past quotas will incur costs.)
KMEANS_SIZE=50 #Maximum number of groups to group the acquired images into

#Labeling
IMG_SHOWN = 30 #Number of total images to show in the labeling Notebook, adjust as necessary
NUM_IMG_ROW = 3 #Number of images to show in a single row, adjust to scale as necessary

#Get_Images.py
DATASIZE=2500 #Number of images to work with at a given time (MEMORY LIMITED)

#Train_Model.py
TRAINING_N_EPOCHS=50 #Maximum number of epochs to train for, currently using early stopping, so unlikely to reach this maximum bound


###Global Constants
RANDOM_STATE = 0 #State for random functions to ensure some replicability
GEE_COUNTRY_DATASET = "USDOS/LSIB_SIMPLE/2017" #GEE dataset containing relevant country bounds for tiling
GEE_IMAGE_SHAPE = 64 #Shape of wanted image
GEE_IMAGE_FORMAT = 'png' #Format of wanted image
GEE_MAX_PIXEL_VALUE = .3000 #Maximum Pixel value of returned GEE image
GEE_IMAGE_TIMEOUT = 15 #Timeout for a given GEE Call
GEE_SATELLITE = "COPERNICUS/S2_SR_HARMONIZED" #Satellite to acquire GEE Images
GEE_FILTERS = [
    'CLOUDY_PIXEL_PERCENTAGE',
    'DARK_FEATURES_PERCENTAGE',
    'THIN_CIRRUS_PERCENTAGE'
] #Filter Names to limit unwanted images
GEE_FILTERS_BOUNDS = [
    20,
    20,
    20
] #Numerical bound to limit unwanted images
GEE_SORT_CATEGORY = 'HIGH_PROBA_CLOUDS_PERCENTAGE' #Desired Sort category
GEE_START_DATE = '2020-09-01' #Determines lower bound of satellite imagery
GEE_END_DATE = '2021-09-01' #Determines upper bound of satellite imagery
CUDA='cuda'

###PATHS
ABSOLUTE_ROOT=pathlib.Path(__file__).resolve().parents[1].as_posix()
IMAGE_DATASETS_ROOT=ABSOLUTE_ROOT+'/Data/Image_Datasets/'
MODEL_ROOT = ABSOLUTE_ROOT+'/Data/Model_Logs/'
GEOJSON_ROOT = ABSOLUTE_ROOT+'/GeoJSONS/'
PREDICTION_ROOT = ABSOLUTE_ROOT+'/Data/Predicted_Datasets/'
STATE_SAMPLING_ROOT = ABSOLUTE_ROOT+'/Data/State_Separated_Datasets/'
GROUPING_ROOT=ABSOLUTE_ROOT+'/Data/Labeling_Data/'
COORDINATES_ROOT=ABSOLUTE_ROOT+'/Data/Coordinates/'
BRICK_BELT=ABSOLUTE_ROOT+'/GeoJSONS/Brick_Belt.geojson'
