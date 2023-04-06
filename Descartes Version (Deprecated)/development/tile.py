"""
Used to access shapefiles and tile over them. 
"""
import geojson
import descarteslabs as dl
import tempfile


def create_general_aoi(location: str, property_name: str, geo_json_path: str="../GeoJSONS/countries.geojson"):
    """
    Return aoi object for the location specified by property_name under the
    properties object of geo_json_path
    """
    with open(geo_json_path) as f:
        file = geojson.load(f)['features']
    
    found = False
    for obj in file:
        location_name = obj['properties'][property_name]
        if location_name == location:
            geotype=obj['geometry']['type']
            location_obj = obj
            found = True
            break
    
    if not found:
        print(f"No matching location with name {location} found in property {property_name} in {geo_json_path}")
    aoi_coords = location_obj['geometry']['coordinates']
    return {
        "type": "Feature",
        "geometry": {
            "type": geotype,
            "coordinates": aoi_coords,
        },
        "properties": {},
    }

def create_country_tiles(tilesize: int=2048, resolution: int=10, location: str="Bangladesh", property_name: str="ADMIN"):
    """
    @param tilesize: side length of one square tile
    @param resolution: per-pixel resolution of the generated proxy object
    @param location: location over which tiles are generated
    @param property_name: name of the key in the geojson property field that countains the location name
    @param geo_json_path: path to the geojson
    
    @ return aoi (dict): dictionary containing geographic data that can easily be turned in DL.Feature 
                            object for plotting
    @return tiles (dl.scenes.DLTile): Geographic grid tiles of equal dimension over country 
    """
    aoi = create_general_aoi(location, property_name)
    tile_generator = dl.scenes.DLTile.iter_from_shape(aoi, 
                                    tilesize=tilesize, 
                                    resolution=resolution,
                                    pad=0)
    
    tile_keys = [tile.key for tile in tile_generator]
    return aoi, tile_keys