Fresh Install Guide


*Install requirements.txt using your preferred Library Manager (pip is preferred as that's what was originally used, pip install -r requirements.txt)

*Open and run the Authenticate Google Earth.ipynb file to register usage of the google earth engine authorization token.  This will need to be re-run on occassion once or twice a month, you'll know because you'll try to run something and it will not allow you to, at which point do this step again.

*Download at least 1 model, the GeoJSONS folder and any relevant datasets from the google drive and place them in their respective files in the data folder and the GeoJSONS folder

*Go to the Configs Directory and generate a keys.py file from the keys_base.py file.  (Duplicate and rename as appropriate)  Find your google maps key in the google console api list and paste into the folder (Model_Logs for models, Prdicted_Datasets for prediction/sampled results csvs, Image_Datasets full image datasets in npy format, Predicted Datasets for batch, Confirmed and Denied files to reference.

*Duplicate and appropriately fill the config_base.py file (constants.py should give a general outline of how and what to fill the parts with) 
	*The Common Variable Section is originally empty and will need to be filled in completely for each respective file to run correctly (Sampling Dataset Name can be empty if running the first batch of files (Prediction/Sampling Step) but will need to be filled with the relevant generated file to proceed with the next steps) 
	*The Advanced Variables section will largely remain the same unless there comes a need to improve runtime/workflow on your personal machine.  The main exception is the DEPLOY_START_FROM variables which should be altered to the appropriate start value in case Predict_Country.py crashes mid-runtime.
	*The Constants Variables section will largely remain the same unless different border datasets are used/better GEE values are found.  The main exception are the GEE_START_DATE and GEE_END_DATE which represent the time periods to pull images from.  Alter as necessary to obtain data from those time periods specifically (NOTE: pulling from a too-small time period may result in a reduced number of images obtained)
	*The PATHS variables section should remain exactly as is unless the directory structure is changed, from the box it should be primed to work as intended but be aware that they may need to change if your file structure changes for any reason.


File Structure and Description

*(DEBUG) - 
	This directory holds a few relevant files to help visualize/test certain files for integrity/shape, should largely not need to be accessed unless something has gone really wrong or you'd like to view a specific dataset's information quickly

*(DEPRECATED) Notebook Versions - 
	Legacy .ipynb files from before the Scripts were finalized, largely exist for reference purposes but will be removed once pipeline is satisfactorially finalized.  There's nothing here that is worth looking at unless new changes need to be made and some reference is required.

*Configs -
	Directory where a few important files are located as well all existing config files for a given subset of tests/datasets.
		-keys.py -
			File containing the Google Maps Api (Reference the google console for quotas/current usage to avoid being charged/overusing data)
		-constants.py - 
			Base file for scripts to reference if there is no referenced config file (also serves as a working example of what a config file should look like)
		-config_base.py - 
			Base file for script creation, should be duplicated and renamed to help differentiate test/batches of datasets

*Data -
	Directory that contains/will be populated with the relevant files
		-Coordinates -
			Contains the coordinate jsons for a given dataset, generated in the Get_Coordinates.py script
		-Image_Datasets - 
			Each dataset contains a metadata.csv file for quick reference and two directories separating positive and negative kiln images in the npy format.  Generated in the Get_Images.py script.
		-Labeling_Data - 
			Each dataset contains a group of 3 files a GroupBatch# file, a LrBatch# file and a HrBatch# file containing respectively the images contained in a given batch of images, the low res images (GEE) and the high res images (Google Maps API). These are generated in the Group_Predictions.py file.  Each dataset also has a Confirmed and Denied File containing what images have been labeled positive and negatively as well as how they've been labeled if positive.  These files are generated in the labeling process in the Labeling.ipynb file
		-Model_Logs -
			Each folder contains a set of checkpoints to reference for testing/evaluating purposes as well as a last and best checkpoint for use in predicting.  Generated in the Train_Model.py script 
		-Predicted_Datasets -
			This Directory contains each complete set of predictions of a given area, generated in the Predict_Country.py script, as well as a set of sampled predictions according to the specified restrictions in the config file, generated in the Sample_Predictions.py script.
		-State_Separated_Datasets -
			This Directory contains each Sampled dataset filtered by it's state of residence, generated in teh Sample_Predictions.py script, largely irrelevant outside of verifying data integrity

*GeoJSONS - 
	Directory containing all of the referenced countries' boders and state borders as well a general area called the Brick_Belt from which to focus our efforts on. Should not need to do much more than find relevant geojsons for areas of interest and placing them within.
	
*Labeling - 
	Home of the labeling app and it's relevant utilities.  After a dataset has gone through the first batch of scripted files (Predicting/Sampling scripts) you'll have a few batches of unlabeled data that can then be labeled here within the Notebook. Ensure the appropriate config file is named in the second cell and it in turn refers to the appropriate sampled dataset. Follow through the notebook until there are no other images to label, save and close the notebook.

*Predicting_Utils -
	Home of additional files for the Predict_Country.py script.  Largely irrelevant unless something specifically needs to be changed, can be interacted instead via the config files' Advanced Variable Sections and Constants section

*Scripts -	
	Home of the scripts grouped generally into three batches the Predicting/Sampling Batch (Predict_Country->Sample_Predictions->Group_Predictions) the Image/Coordinates Batch (Get_Images->Get_Coordinates) and the Model Batch (Train_Model).  Each batch can be run wholesale or each step can be run individually to get certain pieces of data.  Each file can be called with the -h tag to print out some relevant information about the script.


WorkFlow Guide

*Full Pipeline -
	The full pipeline runs all 3 of the batches and begins with an existing model, predicts over a country, samples it's dataset, results in some labeling for the user to do, get those sampled images and their relevant coordinates and is then used (immediately or in the future) to train a new model. The instructions are as follow.
	
	1.) Ensure a config file is generated for this batch and all but the Sampling Dataset Name is filled and valid for any scripts that will be run, this should be done each time before running any script but should largely not change from the initial creation
	
	2.) Open the Scripts directory in the terminal and run the first batch of scripts using 
		python3 Predict_Country.py -f -co YourConfigFileNameHere

	Note that this step is the most likely to crash/be interrupted.  If so do as follows depending on where/how the script was interrupted
	
		2a.) If the Script crashed/stopped in the Predict_Country step
			2aa.)  It can be easily restarted by noting the number of the last tile predicted, filling in the DEPLOY_START_FROM section in the config and running the Predict_Country.py script on it's own and later combining them all together.  Terminal Example is as follows
				python3 Predict_Country.py -co YourConfigFileNameHFileNameHere.py

			2ab.) After all datasets have been generated, head to the Predicted_Datasets folder in the Data Folder and fill the datasets list with the relevant fractured datasets.  Run the Notebook until the 4th cell and ensure you name the file appropriately and that it doesn't coincide with another existing file, then run the cell and note the final name of the dataset.

		2b.) Code is least likely to encounter errors in the Sample_Predictions step but if it does, fill in the SAMPLING_DATASET_NAME section in the relevant config file and ensure the relevant generated files in the Data/Predicted_Datasets and Data/State_Separated_Datasets are removed.  These files will look in accordance with the combined SAMPLING_DATASET_PREFIX and provided dataset name so ensure the appropriate files are removed and not the original full dataset. Code snippet is as follows to reinitiate this step

			python3 Sample_Predictions -co YourConfigFileNameHere.py

		2c.) If this Script has crashed/stopped in the Group_Sampling step ensure that the SAMPLING_DATASET_NAME is filled in the config file and the GROUPING_DATASET_NAME refers to an existing file in the Data/Predicted_Datasets folder.  Once done so, clear the relevant directory in the Data/Labeling_Data folder and reinitiate this step of the script, code snippet is as follows to do so

			python3 Group_Predictions -co YourConfigFileNameHere.py

	3.) Ensure the SAMPLING_DATASET_NAME is filled in your config file and to fill the any empty points for the Labeling, Get_Images.py and Get_Coordinates.py sections.
	
	4.) Open the Labeling Notebook in the Labeling Directory and ensure the appropriate config file is called and label your relevant data until there is none left.
	
	5.) Return to the Scripts directory and begin the Images/Coordinate batch of Scripts. The following code snippet achieves this result
		
		python3 Get_Images -cd -co YourConfigFileNameHere.py

	6.) Finally fill in all relevant sections in the Train_Model in your config file and run the Train_Model.py script as follows

		python3 Train_Model -co YourConfigFileNameHere.py

