import pathlib

###PATHS
ABSOLUTE_ROOT=pathlib.Path(__file__).resolve().parents[1].as_posix()
IMAGE_DATASETS_ROOT=ABSOLUTE_ROOT+'/Data/Image_Datasets/'
MODEL_ROOT = ABSOLUTE_ROOT+'/Data/Model_Logs/'
GEOJSON_ROOT = ABSOLUTE_ROOT+'/GeoJSONS/'
PREDICTION_ROOT = ABSOLUTE_ROOT+'/Data/Predicted_Datasets/'
STATE_SAMPLING_ROOT = ABSOLUTE_ROOT+'/Data/State_Separated_Datasets/'
GROUPING_ROOT=ABSOLUTE_ROOT+'/Data/Labeling_Data/'
COORDINATES_ROOT=ABSOLUTE_ROOT+'/Data/Coordinates/'


###COMMON VARIABLES
#Predict_Country.py
DEPLOY_MODEL_WEIGHTS = 'Upsampled_Train_bipn'
DEPLOY_COUNTRY = 'India'

#Sample_Predictions.py
SAMPLING_DATASET_NAME = 'calm_snake'
SAMPLING_COUNTRY = DEPLOY_COUNTRY
SAMPLING_DATASET_PREFIX = 'FULL_TEST_'

#Group_Predictions.py
GROUPING_DATASET_NAME=SAMPLING_DATASET_PREFIX+SAMPLING_DATASET_NAME

#Labeling
IMG_SHOWN=30
NUM_IMG_ROW=3

#Get_Imaegs.py
GENERATE_DATASET_NAME=GROUPING_DATASET_NAME

#Get_Coordinates.py
COORDINATE_MODEL=DEPLOY_MODEL_WEIGHTS
COORDINATE_DATASET_NAME=GENERATE_DATASET_NAME
COORDINATE_POSTFIX='_FULL_TEST'

#Train_Model.py
TRAINING_MODEL_NAME='FULL_TEST'
TRAINING_DATASET_LIST=[
    'FULL_TEST_calm_panda',
    'FULL_TEST_fiery_pig',
    'FULL_TEST_calm_snake',
    'FULL_TEST_gaudy_snake'
]


###ADVANCED VARIABLES
#Predict_Country.py
DEPLOY_START_FROM=0

#Sample_Predictions.py
MAX_KILN_STATE_SAMPLE=500
MAX_NON_KILN_STATE_SAMPLE=250
MIN_NON_KILN_STATE_SAMPLE=50
RANDOM_SAMPLE=500
OVERSAMPLE_STATES=['F.A.T.A','N.W.F.P','Punjab']
OVERSAMPLE_MODIFIER=1.5
BRICK_BELT=ABSOLUTE_ROOT+'/GeoJSONS/Brick_Belt.geojson'
PREDICTION_CONFIDENCE_BOUNDS=.90

#Group_Predictions.py
BATCH_SIZE=1100
KMEANS_SIZE=50

#Get_Images.py
DATASIZE=2500

#Train_Model.py
TRAINING_N_EPOCHS=50


###Global Constants
RANDOM_STATE = 0
GEE_COUNTRY_DATASET = "USDOS/LSIB_SIMPLE/2017"
GEE_IMAGE_SHAPE = 64
GEE_IMAGE_FORMAT = 'png'
GEE_MAX_PIXEL_VALUE = .3000
GEE_IMAGE_TIMEOUT = 15
GEE_SATELLITE = "COPERNICUS/S2_SR_HARMONIZED"
GEE_FILTERS = [
    'CLOUDY_PIXEL_PERCENTAGE',
    'DARK_FEATURES_PERCENTAGE',
    'THIN_CIRRUS_PERCENTAGE'
]
GEE_FILTERS_BOUNDS = [
    20,
    20,
    20
]
GEE_SORT_CATEGORY = 'HIGH_PROBA_CLOUDS_PERCENTAGE'
GEE_START_DATE = '2020-09-01'
GEE_END_DATE = '2021-09-01'
















