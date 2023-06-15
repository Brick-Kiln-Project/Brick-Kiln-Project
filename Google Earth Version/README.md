# Brick Kiln Project

## Description

The Brick Kiln Project is 

This project is intended to get satellite imagery of predicted kilns across a given country using Google Earth Satellites.  This project contains a full pipeline for training/updating a model from scratch, including a script for obtaining images, sorting/filtering them and preparing them for labeling, a small notebook for labeling the images and an additional script for obtaining coordinates of high-probability kilns in the images and saving/prepping images for the model to train/evaluate.  These scripts are intended to be use after each other to fine tune a model for your needs use, but can also be compartmentalized to get relevant coordinate data, images or simply data on predicted kilns across the requested country.  In order to use the following scripts you'll need to have an existing Google Earth Engine and General Google API Account, which can be obtained for free for personal use or at cost with a professional account although the free versions have been sufficient.  

---
## General Workflow

### Locate Coordinates -

If you only intend to search for kilns in a given country (or other subset of provided google earth dataset) you can simply run the following

`python3 Predict_Country.py -co YourConfigFileNameHere -coor CoordianteFileNameHere`

to quickly run through the country (or otherwise) provided for the DEPLOY_COUNTRY variable in the provided constant file and return the coordinates of any high-probability areas in high-probably kiln images.

### Quick Pipeline - 

If you intend to generate a new dataset for training/evaluating there is a quick set of 3 scripts to quickly add to your available dataset and eventually train.  The following in order will get you up and running

`python3 Predict_Country.py -f -co YourConfigFileNameHere`

This will sequentially predict for kilns across the country, sort them into their relevant state and sample them into the final predicted dataset using the predefined variables in the provided config file.  Ensure that everything within the PREDICT_COUNTRY,SAMPLE_PREDICTIONS and GROUP_PREDICTIONS sections are set to the desired variables

After this, you should head into the Labeling directory and open the Labeling notebook to label the dataset for training.  Ensure the correct config file is provided and the SAPLING_DATASET_NAME is updated to match your desired dataset to label.

`python3 Get_Images -cd -co YourConfigFileNameHere.py`

This will sequentialy separate the labeled data into positive and negative images and dredge through the positive images for the coordinates of the highest predicted areas in the image.  Ensure the correct config file is provided and everything under the GET_IMAGES and GET_COORDINATES sections is set to desired variables.

From here repeat for as many countries/datasets you'd like to make for training/evaluating the model.

`python3 Train_Model -co YourConfigFileNameHere.py`

This will load up all of your supplied labeled data, split it into training/validation and train a new model. Ensure the provided config file is correct and the TRAIN_MODEL section is correct and up to date.

### Full Pipeline - 

The full pipelines consists of a total of 6 scripts, with each individualized section handling a specific task.  These exist separately for two reasons, the first being that it is helpful to jump back in at any point in case of catastrophic failure or other forms of error, usually also having some variables in the config file to cut back to the cutoff point and the second being to allow for creation/recreation of certain data without having to rely on previous sections of the pipeline (e.g. sample separate country data from the same predicted dataset).  Below is each script and how it can be called.
	
`python3 Predict_Country.py [-f,-full] [-h,-help] [-co,-config] ConfigFile [-c,-country] CountryName [-m,-model] ModelName [-coor,-coordinates] CoordinateFileName`

- [-f,-full] - Sets the script to sequentially run Predict_Country, Sample_Predictions and Group_Predictions one after the other, feeding the relevant data forward
- [-h,-help] - Prints a small help text on the function before ending
- [-co,-config] - Sets the config file to pull variables and relevant flags from, if ommited defaults to using the constants config file (NOT RECOMMENDED)
- [-c,-countru] - Manually sets the country to tile and predict, useful if you don't want to generate a new config file or for one-of dataset creation (the same applies for any manually set variable)
- [-m,-model] - Manually sets the model to use to predict the images.
- [-coor,-coordinates] - Sets a flag to additionally return the coordinates of high-confidence points in images and saves them to the provided file.

This Script tiles an entire country and predicts over each tile image returning a completely detailed dataset containing relevant information on the location and accuracy of the prediction in the Data/Predicted_Datasets/ directory.

**Note that this step is the most likely to crash/be interrupted.  If so do as follows depending on where/how the script was interrupted**

If the Script crashed/stopped it can be easily restarted by noting the number of the last tile predicted, filling in the DEPLOY_START_FROM section in the relevant config file and re-running the Predict_Country.py script and later combining them all together.

After all datasets have been generated, head to the Data/Predicted_Datasets/ directory and open the Combine CSV notebook.  Fill the datasets list with the relevant fractured datasets.  Run the Notebook up until the 4th cell and replace the string with the desired name for the complete dataset and that it doesn't coincide with another existing file, then run the cell and note the final name of the dataset.

`python3 Sample_Predictions [-h,-help] [-co,-config] ConfigFile [-c,-country] CountryName [-p,-prefix] Prefix [-d,-dataset] DatasetName`

- [-h,-help] - Prints a small help text on the function before ending
- [-co,-config] - Sets the config file to pull variables and relevant flags from, if ommited defaults to using the constants config file (NOT RECOMMENDED)
- [-c,-country] - Manually sets the country to sample by state of, requires a relevant geojson file within the geojson directory
- [-p,-prefix] - Manually sets the prefix to name the resulting sampled dataset
- [-d,-dataset] - Manually sets the complete predicted datset to sample

This script iterates through the provided complete dataset and samples them by state and then by filters them out given the state and it's size in proportion to the rest of the state and returns a sampled dataset in the Data/Predicted_Datasets/ directory.  These parameters can be changed individually in the ADVACED_SECTION SAMPLE_PREDICTIONS.

**If this step crashes or is otherwise interrupted do as folows**

If this fails, ensure the SAMPLING_DATASET_NAME section in the config file is correct and delete the relevant generated files in the Data/Predicted_Datasets and Data/State_Separated_Datasets.  These files will look in accordance with the combined SAMPLING_DATASET_PREFIX and provided dataset name so ensure the appropriate files are removed and **not the original full dataset**.  Reinitiate with the same flags to continue

`python3 Group_Predictions [-h,-help] [-co,-config] ConfigFile [-d,-dataset] DatasetName`

- [-h,-help] - Prints a small help text on the function before ending
- [-co,-config] - Sets the config file to pull variables and relevant flags from, if ommited defaults to using the constants config file (NOT RECOMMENDED)
- [-d,-dataset] - Manually sets the sampled dataset name to group

This script groups up the entire sampled dataset into smaller/loadable batches for labeling as well as saving the low and high resolution images of said images to make labeling faster in the Data/Labeling_Data/ directory.

**If this step crashes or is otherwise interrupted do as folows**

If the script has crashed/stopped ensure that the SAMPLING_DATASET_NAME is filled in the config file and the GROUPING_DATASET_NAME refers to an existing file in the Data/Predicted_Datasets folder.  Once done so, clear the generated directory in the Data/Labeling_Data folder and reinitiate this step of the script.

### **LABEL AT THIS POINT IN THE LABELING DIRECTORY**

`python3 Get_Images [-h,-help] [-cd,-coordinates] [-co,-config] ConfigFile [-d,-dataset] DatasetName`

- [-h,-help] - Prints a small help text on the function before ending
- [-cd,-coordinates] - sets a flag to sequentially run Get_Images and Get_Coordinates one after the other feeding relevant data forward
- [-co,-config] - Sets the config file to pull variables and relevant flags from, if ommited defaults to using the constants config file (NOT RECOMMENDED)
- [-d,-dataset] - Manually sets the Labeled Dataset to get images from

Gets all of the relevant images from the provided Labeled Dataset Name and separates them into positive and negative folders with a larger metadata for global control.  The directory is made in Data/Image_Datasets/ and populates a generated dataset with the dataset name.

`python3 Train_Model [-h,-help] [-co,-config] ConfigFile [-m,-model] ModelName`

- [-h,-help] - Prints a small help text on the function before ending
- [-co,-config] - Sets the config file to pull variables and relevant flags from, if ommited defaults to using the constants config file (NOT RECOMMENDED, THIS SCRIPT TAKES DATA TO TRAIN/EVALUTE FROM THE CONFIG FILE)
- [-m,-model] - Manually sets the name of the model to save

This script loads up the inputed datasets in the config file and trains up a model leaving it in the Data/Model_Logs/ directeory populating a directory with it's name with the final model and a few checkpoints for debugging/logging.

---
## Project Structure & Description

Brick-Kiln-Project\
|-Descartes Version (Deprecated)\
|-Google Earth Version\
| |-(DEBUG)\
| |-(DEPRECATED)\
| |-Configs\
| | |-config_base.py\
| | |-constants.py\
| | |-keys_base.py\
| |-Data\
| | |-Coordinates\
| | |-Image_Datasets\
| | | |-Example_Dataset\
| | | | |-negatives\
| | | | |-positives\
| | | | |-metadata.csv\
| | |-Labeling_Data\
| | | |-Example_Dataset\
| | |-Model_Logs\
| | | |-Example_Dataset\
| | | | |-checkpoings\
| | |-Predicted_Datasets\
| | | |-Combine CSV.ipynb\
| | |-State_Separated_Datasets\
| |-GeoJSONS\
| |-Labeling\
| | |-Labeling.ipynb\
| |-Predicting_Utils\
| |-Scripts\
| | |-Get_Coordinates.py\
| | |-Get_Images.py\
| | |-Group_Predictions.py\
| | |-Predict_Country.py\
| | |-Sample_Predictions.py\
| | |-Train_Model.py\
| |-ESPCN_x4.pb\
| |-requirements.txt

- (DEBUG) -
  
	This directory holds a few relevant files to help visualize/test certain files for integrity/shape, should largely not need to be accessed unless something has gone really wrong or you'd like to view specific dataset information quickly

- (DEPRECATED) Notebook Versions -
  
	Legacy .ipynb files from before the Scripts were finalized, largely exist for reference purposes but will be removed once pipeline is satisfactorially finalized.  There's nothing here that is worth looking at unless new changes need to be made and some reference is required.

- Configs -

    Directory where a few important files are located as well all existing config files for a given subset of tests/datasets.
  
  - keys.py - File containing the Google Maps Api (Reference the google console for quotas/current usage to avoid being charged/overusing data)
  - constants.py - Base file for scripts to reference if there is no referenced config file (also serves as a working example of what a config file should look like)
  - config_base.py - Base file for script creation, should be duplicated and renamed to help differentiate test/batches of datasets

- Data -
  
    Directory that contains/will be populated with the relevant files
    - Coordinates -
	Contains the coordinate jsons for a given dataset, generated in the Get_Coordinates.py script
	- Image_Datasets - 
	Each dataset contains a metadata.csv file for quick reference and two directories separating positive and negative kiln images in the npy format.  Generated in the Get_Images.py script.
	- Labeling_Data - 
	Each dataset contains a group of 3 files a GroupBatch# file, a LrBatch# file and a HrBatch# file containing respectively the images contained in a given batch of images, the low res images (GEE) and the high res images (Google Maps API). These are generated in the Group_Predictions.py file.  Each dataset also has a Confirmed and Denied File containing what images have been labeled positive and negatively as well as how they've been labeled if positive.  These files are generated in the labeling process in the Labeling.ipynb file
    - Model_Logs -
    Each folder contains a set of checkpoints to reference for testing/evaluating purposes as well as a last and best checkpoint for use in predicting.  Generated in the Train_Model.py script 
	- Predicted_Datasets -
	This Directory contains each complete set of predictions of a given area, generated in the Predict_Country.py script, as well as a set of sampled predictions according to the specified restrictions in the config file, generated in the Sample_Predictions.py script.
	- State_Separated_Datasets -
	This Directory contains each Sampled dataset filtered by it's state of residence, generated in teh Sample_Predictions.py script, largely irrelevant outside of verifying data integrity

- GeoJSONS -
  
	Directory containing all of the referenced countries' boders and state borders as well a general area called the Brick_Belt from which to focus our efforts on. Should not need to do much more than find relevant geojsons for areas of interest and placing them within.
	
- Labeling -
  
	Home of the labeling app and it's relevant utilities.  After a dataset has gone through the first batch of scripted files (Predicting/Sampling scripts) you'll have a few batches of unlabeled data that can then be labeled here within the Notebook. Ensure the appropriate config file is named in the second cell and it in turn refers to the appropriate sampled dataset. Follow through the notebook until there are no other images to label, save and close the notebook.

- Predicting_Utils -
  
	Home of additional files for the Predict_Country.py script.  Largely irrelevant unless something specifically needs to be changed, can be interacted instead via the config files' Advanced Variable Sections and Constants section

- Scripts -
   
	Home of the scripts grouped generally into three batches the Predicting/Sampling Batch (Predict_Country->Sample_Predictions->Group_Predictions) the Image/Coordinates Batch (Get_Images->Get_Coordinates) and the Model Batch (Train_Model).  Each batch can be run wholesale or each step can be run individually to get certain pieces of data.  Each file can be called with the -h tag to print out some relevant information about the script.

---
## Fresh Install Guide

- Install requirements.txt using your preferred Library Manager (pip is preferred as that's what was originally used, pip install -r requirements.txt)

- Go to your Google Earth Console and generate a service account, note the account name and download the json key to /Configs/ Directory.

- Download at least 1 model, the GeoJSONS folder and any relevant datasets from the storage and place them in their respective files in the data folder and the GeoJSONS folder

- Go to the Configs Directory and generate a keys.py file from the keys_base.py file.  (Duplicate and rename as appropriate)  Find your google maps key in the google console api list and paste into the file alongside your google earth service account name (Model_Logs for models, Prdicted_Datasets for prediction/sampled results csvs, Image_Datasets full image datasets in npy format, Predicted Datasets for batch, Confirmed and Denied files to reference.

- Duplicate and appropriately fill the config_base.py file (constants.py should give a general outline of how and what to fill the parts with) 
  - The Common Variable Section is originally empty and will need to be filled in completely for each respective file to run correctly (Sampling Dataset Name can be empty if running the first batch of files (Prediction/Sampling Step) but will need to be filled with the relevant generated file to proceed with the next steps)
  - The Advanced Variables section will largely remain the same unless there comes a need to improve runtime/workflow on your personal machine.  The main exception is the DEPLOY_START_FROM variables which should be altered to the appropriate start value in case Predict_Country.py crashes mid-runtime.
  - The Constants Variables section will largely remain the same unless different border datasets are used/better GEE values are found.  The main exception are the GEE_START_DATE and GEE_END_DATE which represent the time periods to pull images from.  Alter as necessary to obtain data from those time periods specifically (NOTE: pulling from a too-small time period may result in a reduced number of images obtained)
  - The PATHS variables section should remain exactly as is unless the directory structure is changed, from the box it should be primed to work as intended but be aware that they may need to change if your file structure changes for any reason.
