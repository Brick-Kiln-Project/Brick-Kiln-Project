import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

import pickle as pkl
import ee
ee.Initialize()
from time import sleep
import sys
from math import floor
sys.path.append("../Configs/")
#import constants
from global_func import GEELoadImage, masks2clouds

class TileDataset(Dataset):
    
    def __init__(self, tile_geometry, idx,country,constants, rgb_only=True):
        sleep(idx%60)
        GEE_SATELLITE,GEE_START_DATE,GEE_END_DATE,GEE_FILTERS,GEE_FILTERS_BOUNDS,GEE_MAX_PIXEL_VALUE,GEE_IMAGE_FORMAT,GEE_IMAGE_SHAPE=constants
        #Tile current tile into 1024 total tiles, first try is likely to fail, second exists as redundancy
        try:
            self.subtile_geometries = ee.Geometry(tile_geometry['geometry']).coveringGrid("EPSG:3857",700).getInfo()
        except:
            self.subtile_geometries = ee.Geometry(tile_geometry['geometry']).coveringGrid("EPSG:3857",700).getInfo()
        
        #Initialize class variables
        self.img_data=[None]*1024
        self.idx=[None]*1024
        batches=ee.Geometry(tile_geometry['geometry']).coveringGrid("EPSG:3857",2800).getInfo()['features']
        #Initialize and fill the image collection with the satellite and it's filters
        lowres_collection=ee.ImageCollection(GEE_SATELLITE).filterDate(GEE_START_DATE,GEE_END_DATE)
        for filters,bounds in zip(GEE_FILTERS,GEE_FILTERS_BOUNDS):
            lowres_collection=lowres_collection.filter(ee.Filter.lte(filters,bounds))
        lowres_collection=lowres_collection.map(masks2clouds)
        for bindex,batch in enumerate(batches):
            startbx=bindex%8
            startby=floor(bindex/8)
            batch_subtiles=[]
            for y in range(4):
                batch_subtiles.extend(list(range(((128*startby)+(32*y))+(4*startbx),((128*startby)+(32*y))+(4*(startbx+1)))))
                
            try: 
                lowRes_collection=lowres_collection.filterBounds(ee.Geometry(batch['geometry']))
                image=lowRes_collection.mean()
                buff=[GEE_MAX_PIXEL_VALUE,GEE_IMAGE_FORMAT,GEE_IMAGE_SHAPE*4]
                data=GEELoadImage(image,batch['geometry'],buff,rgb_only)
            except Exception as e:
                print(e)
                data=None
                
            #Iterate through the subtiles and load the image into it's respective class variable
            for index,ind in enumerate(batch_subtiles):
                cur=idx*1024+ind
                if data is None:
                    self.img_data[ind]=data
                else:
                    startx=(index%4)*64
                    starty=(3-floor(index/4))*64
                    self.img_data[ind]=data[starty:starty+64,startx:startx+64,:]
                self.idx[ind]=cur
    
    def __getitem__(self, index):
        if self.img_data[index] is not None:
            img_slice = self.img_data[index]
            tile_geometry = self.subtile_geometries['features'][index]
            idx=self.idx[index]
            return pkl.dumps(tile_geometry), Tensor(img_slice).float(), idx
        else:
            return 'False','False','False'
        
    
    def __len__(self):
        return len(self.subtile_geometries['features'])
    
    
def get_tile_DataLoader(tile_geometry, idx,country,constants, rgb_only=True, num_workers=4, batch_size=1):
    """ Returns the dataloader for iterating over tile data."""
    dset = TileDataset(tile_geometry, idx,country,constants, rgb_only=rgb_only)
    return DataLoader(dset, num_workers=num_workers, batch_size=batch_size)
        
    

