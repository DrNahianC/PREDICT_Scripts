This is the directory to analyse the EEG dataset. 

### Pre-requirements to Run Script

## EEGLAB AND FIELDTRIP
You will need MATLAB, EEGLAB and FIELDTRIP to run these scripts. These should be kept in the 
"eeglab_fieldtrip" folder. The current scripts call on "eeglab2019_1" and "fieldtrip-20200215", so 
if you have another version, you will need to update the script on Lines 19-28 of Sensor_PAF_Pipeline.mat
, Lines 23-32 of Manual_PAF_Pipeline.m and Lines 40-42 of Automated_PAF_Pipeline.m to the version
that you are using. e.g. if you are using "eeglab2023.0", then change all references of "eeglab2019_1"
to "eeglab2023.0". 

## Raw Data
You can download all the EEG data from https://openneuro.org/datasets/ds005486/versions/1.0.0 and 
store this in the "PREDICT_EEG_Raw_Data" directory

## Channels and Templates
In "channel_info", we have already uploaded details of the 63 channels for the EEG recording. We have also uploaded
the "neighbours" template which contains the neighbouring channels of each channel needed to 
interpolate any excluded electrodes. 

Lastly, in "tempalte" we have uploaded "SarahTemplate_2" which is the sensorimotor
component template (obtained from Author Sarah Margerison) used to choose the "winning sensorimotor
component" in the automated ICA pipeline. 

## Output Directories
We have already included the Output directories for each of the three pipelines

### Using the Three different Scripts
To analyse the EEG data, please start at "Sensor_PAF_Pipeline.m" which is the backbone for the 
Automated and Manual component selection pipleines. It contains the script for the raw data loading (i.e the brain vision files), bad channel selection, bad epoch selection, ICA to remove ocular 
artefacts and interpolation of channels. The output of this is used to calculate the sensorimotor ROI PAF in the main paper. All data is saved in the Output Folder directory

The Automated and manual component selection scripts start at the point after bad epochs are removed
in fieldtrip from the Sensor_PAF_Pipeline script. Specifically, it calls on data from the "Output"
data directory, and loads "data_rejected" which is the fieldtrip EEG data structure post-bad
epoch removal. Thus, these scripts will only work once the Sensor_PAF_Pipeline processing is complete.
The job of these scripts is to isolate the sensorimotor alpha component, which is done manually
(Manual_PAF_Pipeline) or automated (Automated_PAF_Pipeline). As the Manual_PAF_Pipeline (used in the
main analysis for the paper) requires a user's input, we have uploaded the chosen 
sensorimotor alpha components on OSF for users of this script to compare their choices with the
choices made in our paper. 
