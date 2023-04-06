import torch
from torch.utils.data import Dataset, DataLoader
from descarteslabs.workflows.models.exceptions import JobTimeoutError
import numpy as np
import descarteslabs as dl
import descarteslabs.workflows as wf
import json
import pickle as pkl

import sys
sys.path.append("../")
import constants

class TileDataset(Dataset):
    
    def __init__(self, tile_key, rgb_only=False, imgs_per_dim=32):
        s2 = wf.ImageCollection.from_id(
            constants.S2_PRODUCT,
            start_datetime=constants.S2_START_DATE,
            end_datetime=constants.S2_END_DATE,
        )
        bands = constants.RGB_BANDS if rgb_only else constants.ALL_BANDS
        s2_bands = s2.pick_bands(bands)
        s2_bands = s2_bands.filter(lambda img: img.properties["cloud_fraction"]<constants.S2_CLOUD_FRACTION)
        s2_bands = s2_bands.median(axis="images")
        
        self.s2_proxy = s2_bands
        
        # set up subtile names
        print(f"Tile key received in Dataset object: {tile_key}")
        self.imgs_per_dim = imgs_per_dim
        self.tile_key = tile_key
        self.tile = dl.scenes.DLTile.from_key(tile_key)
        self.subtile_keys = [tile.key for tile in self.tile.subtile(imgs_per_dim)]
        self.geometry=[tile.geometry for tile in self.tile.subtile(imgs_per_dim)]
        
        # download data for the entire tile, time it
        tile_area = dl.scenes.DLTile.from_key(self.tile_key)
        self.img_data=self.s2_proxy.compute(tile_area,progress_bar=False).ndarray
    
    def __getitem__(self, idx):
        
        if idx > len(self.subtile_keys):
            raise FormatError(f"Idx {idx} is not possible for {len(self.subtile_keys)} subtiles.")
        
        row = (idx // self.imgs_per_dim) * constants.S2_TILESIZE
        col = (idx % self.imgs_per_dim) * constants.S2_TILESIZE
        if self.img_data is not None:
            img_slice = np.array(self.img_data.data[:, row:row + constants.S2_TILESIZE, col:col + constants.S2_TILESIZE])
            subtile_key = self.subtile_keys[idx]
            geometry=self.geometry[idx]
            return subtile_key, torch.Tensor(img_slice), pkl.dumps(geometry)
        else:
            return 'False','False','False'
        
    
    def __len__(self):
        return len(self.subtile_keys)
    
    
def get_tile_DataLoader(tile_key, rgb_only=False, imgs_per_dim=32, num_workers=4, batch_size=1):
    """ Returns the dataloader for iterating over tile data."""
    dset = TileDataset(tile_key, rgb_only=rgb_only, imgs_per_dim=imgs_per_dim)
    return DataLoader(dset, num_workers=num_workers, batch_size=batch_size)
        
    

