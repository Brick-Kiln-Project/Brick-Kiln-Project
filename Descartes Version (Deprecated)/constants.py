"""
Defines the constants to be used across the BK-Winter-2022 directory.
"""
import numpy as np

RGB_BANDS = "red green blue"

ALL_BANDS = "coastal-aerosol blue green red red-edge red-edge-3 red-edge-4 nir swir1 swir2"

S2_TILESIZE = 64
HR_TILESIZE = 640

S2_RESOLUTION = 10
S2_PAD_INDIA = 40
S2_PAD_BANGLADESH= 50

HI_RES_START_DATE = '2018-11-01'
HI_RES_END_DATE = '2020-03-01'
HI_RES_PRODUCT = "airbus:oneatlas:spot:v2"

HR_TILESIZE=640
HR_PAD_INDIA = 400
HR_RESOLUTION = 1

S2_START_DATE = '2021-11-01'
S2_END_DATE = '2022-06-01'

S2_CLOUD_FRACTION = 0.1
S2_PRODUCT = "esa:sentinel-2:l2a:v1"


RANDOM_STATE = 0

DEPLOY_MODEL_WEIGHTS = 'b_i_hsi_usp_usn_prod_weights'

PRODUCT_ID = "kiln-detection-india_v2.0"

TASKS_DOCKER_IMAGE = "us.gcr.io/dl-ci-cd/images/tasks/public/py3.7:v2021.12.20-6-g63e7ba7c"