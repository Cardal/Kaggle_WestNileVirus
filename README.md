Kaggle "West Nile Virus" competition, April - June 2015
=======================================================

Kaggle username "Cardal"
------------------------

This file explains how to run the WNV_Cardal.py code, which generates the winning model in
the Kaggle WNV competition.

**System and Dependencies**
 
The code is written in python. I used Python 2.7.7 on Windows 7 Professional.
The following python libraries are used by the script:
numpy (1.8.1), scipy (0.14.0), pandas (0.14.0), sklearn (0.15.2)
   
**Files**
 
Copy the WNV_Cardal.py script and the SETTINGS.json files to a directory on your system. 
This directory should also contain the training and test data files and the sample submission file. 
   
**Settings**
 
The SETTINGS.json file contains the following parameters:

* "INPUT_DIR": "."      - name of folder with the train.csv, test.csv and sampleSubmission.csv files, as downloaded from Kaggle

* "SUBMISSION_DIR": "." - name of folder in which the submission file with the results will be saved
   
**Execution**

The code should be run from within the directory that contains the script (i.e., you should "cd" into
the directory that contains "WNV_Cardal.py" and the input files).
Simply run "python Python_Cardal.py" - this will read the input files, create several features,
compute prediction probabilities and write the results to a submission file called "wnv_cardal_final.csv".
The execution may take several hours, due to the computation of "nearby multirow counts".