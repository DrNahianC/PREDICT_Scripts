### Pre-requirements 
This contains the scripts for the Machine Learning Analysis. To use these, you will need a version of
Python 3 and a Python IDE (Spyder was used in the current paper).You will also need R to run the growth mixture models. 

### Using the Scripts
"preprocessing_train.py" calls on "Train/Pain_Diary_Data_train.xlsx" to preprocess the raw training set
pain diary data with the output being "Train/df_chew_filled_train.csv" and "Train/df_yawn_filled_train.csv"

"preprocessing_test.py" calls on Test_shuffled/Pain_Diary_Data_Test_Shuffled.xlsx to process the raw 
test set pain diary data with the output being "Test_shuffled/df_chew_filled_test" and 
"Test_shuffled/df_yawn_filled_test"

"R_Script_LGM.R" calls on "Train/df_chew_filled_train.csv", "Train/df_yawn_filled_train.csv"
, "Test_shuffled/df_chew_filled_test.csv" and "Test_shuffled/df_yawn_filled_test.csv" to run
the growth mixture modelling using data from the first 7 days. You will need to set your working 
directory to wherever you have saved this folder on your computer

"shuffle_ID.py" calls on "Test_unshuffled/Pain_Diary_Data_Test_Unshuffled.xlsx" to shuffle the IDs
of test set participants

"parameter_tuning.py" defines the classifier tuner that contains the necessary tools to 
run the machine learning scripts  

"ML_classification_PAF_CME" runs the machine learning scripts on the training set and establishes
predicted pain labels for the test set

"compare_results_PAF_CME" uses the predicted pain labels and comapres against the actual
pain labels to determine model performance 

