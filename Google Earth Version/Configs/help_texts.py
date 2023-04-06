PREDICTION_TEXT="""Predict_Country.py has 2 modes, a fully scripted mode going through Predict_Country.py->Sample_Predictions.py->Group_Predictions.py using variables set in the relevant config file (constants.py as default) and a single mode with callable arguments to set without altering a config file.
            
-f,-full: enters a fully scripted mode running through Predict_Country.py->Sample_Predictions.py->Group_Predictions.py.  Takes up to one more argument (-co,-config) with the relevant config file. Absence defaults to single mode.
-co, -config: submits a config file to load in Google Earth Version/Configs/ with base constants. Absence defaults to using constants.py.
-c, -country: submits a country to run tiling on, must exist within the Config file's GEE_COUNTRY_DATASET. Only usable in single mode. Absence defaults to config file DEPLOY_COUNTRY
-m, -model: submits a model to use to predict with in Google Earth Version/Data/Model Logs/. Only usable in single mode. Absence defaults to config file DEPLOY_MODEL_WEIGHTS
"""

SAMPLING_TEXT="""Sample Prediction takes up to 4 arguments with their paired variables (-co,-config), (-country,-c), (-prefix,-p) and/or (-dataset, -d) each being strings.
    -co, -config:  submits a config file to load in Google Earth Version/Configs/ with base constants. Absence defaults to using constants.py.
    -p, -prefix: submits a string to attach to the front of the resulting datasets to help distinguish, must be alphanumeric characters and/or the followng characters ['-','_',' ']. Absence defaults to config file SAMPLING_DATASET_PREFIX
    -d, -dataset: submits a dataset referencing a parent dataset in Data/Predicted_Datasets that must not have an another existing prefix + dataset file in the same folder, if it does, user must confirm an overwrite or change argument variables. Absence defaults to config file SAMPLING_DATASET_NAME.
    -c, -country: submits a country which should reference an existing GeoJSON state file within the Google Earth Version/GeoJSONS/ folder and a nonexistent file inside Data/State_Separated_Datasets/ with the prefix and dataset.  If a folder alerady exists, you must confirm it's overwrite or change the argument variables. Absence defaults to config file SAMPLING_COUNTRY
"""

GROUPING_TEXT="""Group Predictions takes up to 2 arguments with their paired variables (-co,-config) and (-d,-dataset), each being strings.
    -co, -config:  submits a config file to load in Google Earth Version/Configs/ with base constants. Absence defaults to using constants.py.
    -d, -dataset: submits a dataset referencing a sampled dataset in Data/Predicted_Datasets. Absence defaults to config file GROUPING_DATASET_NAME
"""

IMAGES_TEXT="""Get Images has two modes, a fully scripted mode going from Get_Images.py->Get_Coordinates.py using variales set in the relevant config file (constants.py as default) and a single mode with up to 2 arguments (-co,-config) and (-d,-dataset).
    -cd,-coordinates: enters a fully scripted mode running from Get_Images.py->Get_Coordinates.py. Takes up to one more argument (-co,-config) with the relevant config file. Absence defaults to single mode
    -co, -config:  submits a config file to load in Google Earth Version/Configs/ with base constants. Absence defaults to using constants.py.
    -d, -dataset: submits a dataset folder referencing a set of files in the folder located at Google Earth Version/Data/Labeling Data/. Absence defaults to config file GENERATE_DATASET_NAME
"""

COORDINATE_TEXT="""Get Coordinates takes up to 4 arguments (-co,-config), (-m,-model), (-dataset,-d) and/or (-p,-postfix) each being strings.
    -co, -config:  submits a config file to load in Google Earth Version/Configs/ with base constants. Absence defaults to using constants.py.
    -p, -prefix: submits a string to attach to the end of the resulting coordinate file to help distinguish, must be alphanumeric characters and/or the followng characters ['-','_',' ']. Absence defaults to config file COORDINATE_POSTFIX
    -m, -model: submits a model to use to predict with in Google Earth Version/Data/Model Logs/. Absence defaults to config file COORDINATE_MODEL
    -d, -dataset: submits a dataset folder referencing a set of files in the folder located at Google Earth Version/Data/Labeling Data/. Absence defaults to config file GENERATE_DATASET_NAME
"""

TAIN_TEXT="""Train Model takes up to two arguments, (-co,-config), (-m,-model) each being strings. This script makes extensive use of the supplied config file so ensure the right config file is loaded.
    -co, -config:  submits a config file to load in Google Earth Version/Configs/ with base constants. Absence defaults to using constants.py.
    -m, -model: submits a name to call the model once it is finished training

"""



