#Local Files
#import constants

#Libraries
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter, Retry
from requests import Session
from io import BytesIO
import numpy as np
from cv2 import dnn_superres
import torch

import ee
ee.Initialize()

def GEELoadImage(image,geometry,constants,rgb_only=True):
    #Initialize 3 or 10 band list
    if rgb_only:
        bands=['B4','B3','B2']
    else:
        bands=['B1','B4','B3','B2','B5','B7','B8A','B8','B11','B12']
    
    GEE_MAX_PIXEL_VALUE,GEE_IMAGE_FORMAT,GEE_IMAGE_SHAPE=constants
    #Query gee for the relevant image, attempt 5 times and convert it into an np array of shape (constants.GEE_IMAGE_SHAPE,constants.GEE_IMAGE_SHAPE,len(bands))
    try:
        #Initialize request and it's parameters
        thumbUrl=image.getThumbUrl({
            'region':geometry,
            'bands':bands,
            'min':0,
            'max':GEE_MAX_PIXEL_VALUE,
            'format':GEE_IMAGE_FORMAT,
            'crs':'EPSG:3857',
            'dimensions':str(GEE_IMAGE_SHAPE)+'x'+str(GEE_IMAGE_SHAPE),
        })
        response=None
        #Initialize request handler and exponential backoff in case of unexpected server error (ensures some degree of redundancy and avoid overwhelming gee)
        with Session() as s:
            retries = Retry(
                total=1,
                backoff_factor=0.5,
                status_forcelist=[429, 443, 500, 502, 503, 504]
            )
            s.mount(thumbUrl, HTTPAdapter(max_retries=retries))

            #Make and await response from gee, timeout after 15 seconds
            response = s.get(thumbUrl,timeout=50)
        
        #Change bytes response to a PIL RGB image and then to a numpy array, ensure shape integrity, else return None
        with Image.open(BytesIO(response.content)) as img:
            img = np.array(img.convert('RGB'))
            if img.shape != (GEE_IMAGE_SHAPE,GEE_IMAGE_SHAPE,len(bands)):
                return None
    
    #Catch all relevant exceptions and print relevant error text 
    except ee.EEException as e:
        print("Earth Engine Failure",e)
        return None
    except UnidentifiedImageError as p:
        print("Bad GEE Image Response",response.content)
        return None
    except Exception as e:
        print('Unknown Exception',e)
        return None

    return img

def LoadUpsamplingModel(constants):
    with torch.no_grad():
        sr=dnn_superres.DnnSuperResImpl.create()
        path=constants.ABSOLUTE_ROOT+"/ESPCN_x4.pb"
        sr.readModel(path)
        sr.setModel('espcn',4)
        return sr
    
def masks2clouds(image):
    qa = image.select('QA60')

    ##Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    #Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))

    return image.updateMask(mask).divide(10000);

