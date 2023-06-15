"""
Used to access shapefiles and tile over them. 
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append("../Configs/")
"""
import constants
"""
import keys
import ee
from math import ceil

service_account=keys.googleEarthAccount
credentials = ee.ServiceAccountCredentials(service_account,'../Configs/brick-kiln-project-d44b06c94881.json')
ee.Initialize(credentials)

def create_geometry_collection(location: str, geometry_collection: str):
    """
    Return the Google Earth feature collection of the location in question using the provided geometry collection.
    """
    #Search for the country in the collection, return None on failure
    country = ee.FeatureCollection(geometry_collection).filter(ee.Filter.eq('country_na',location))
    
    if country.geometry().getInfo()['coordinates']:
        return country
    else:
        print('No matching location:',location, 'found in the current collection:',geometry_collection,'Try a different combination')
        return None
    

def create_country_tiles(constants,location: str, geometry_collection: str):
    """
    @param location: location over which tiles are generated
    @param geometry_collection: Google Earth geometry collection to search for location in
    
    @return tile_geometry: tile geometry 
    """
    #generate geometry collection
    country = create_geometry_collection(location, geometry_collection)
    
    #Ensure valid country was found
    if country is None:
        return None,None
    
    #create a grid from within the geometry of tiles and batch them into groups of maximum 5000 (google earth limitations at our pay-level) 
    tgeo=country.geometry().coveringGrid("EPSG:3857",22400)
    tgeoList=[]
    for x in range(ceil(tgeo.size().getInfo()/5000)):
        tgeoBatch=tgeo.toList(5000,5000*x).getInfo()
        tgeoList.extend(tgeoBatch)
        
    tile_geometries = tgeoList[constants.DEPLOY_START_FROM:]
    return tile_geometries,country