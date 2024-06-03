import pandas as pd
import numpy as np

# read Pain_Diary_Data_Test_Unshuffled.xlsx file from Test_unshuffled folder, sheet_name='Pain_Chew_AM'
PD_chewAM_df = pd.read_excel('Test_unshuffled/Pain_Diary_Data_Test_Unshuffled.xlsx', sheet_name='Pain_Chew_AM')
PD_chewPM_df = pd.read_excel('Test_unshuffled/Pain_Diary_Data_Test_Unshuffled.xlsx', sheet_name='Pain_Chew_PM')
PD_yawnAM_df = pd.read_excel('Test_unshuffled/Pain_Diary_Data_Test_Unshuffled.xlsx', sheet_name='Pain_Yawn_AM')
PD_yawnPM_df = pd.read_excel('Test_unshuffled/Pain_Diary_Data_Test_Unshuffled.xlsx', sheet_name='Pain_Yawn_PM')

# shuffle the ID column for each dataframe using np.random.shuffle, keep track of the order of the shuffled IDs by creating a new dataframe that records the original ID and the shuffled ID
ID = PD_chewAM_df['ID'].values
ID_shuffled = np.random.permutation(ID)
ID_df = pd.DataFrame({'ID': PD_chewAM_df['ID'].values, 'ID_shuffled': ID_shuffled})

# save the shuffled ID dataframe to a csv file
ID_df.to_csv('Test_unshuffled/ID_df.csv', index=False)

# replace each dataframe's ID column with the shuffled ID column
PD_chewAM_df['ID'] = ID_shuffled
PD_chewPM_df['ID'] = ID_shuffled
PD_yawnAM_df['ID'] = ID_shuffled
PD_yawnPM_df['ID'] = ID_shuffled

# sort the shuffled dataframes by ID
PD_chewAM_df.sort_values(by=['ID'], inplace=True)
PD_chewPM_df.sort_values(by=['ID'], inplace=True)
PD_yawnAM_df.sort_values(by=['ID'], inplace=True)
PD_yawnPM_df.sort_values(by=['ID'], inplace=True)

# save the shuffled dataframes to one xlsx file with different sheet names
with pd.ExcelWriter('Test_shuffled/Pain_Diary_Data_Test_Shuffled.xlsx') as writer:
    PD_chewAM_df.to_excel(writer, sheet_name='Pain_Chew_AM', index=False)
    PD_chewPM_df.to_excel(writer, sheet_name='Pain_Chew_PM', index=False)
    PD_yawnAM_df.to_excel(writer, sheet_name='Pain_Yawn_AM', index=False)
    PD_yawnPM_df.to_excel(writer, sheet_name='Pain_Yawn_PM', index=False)

# record the dropped IDs in a csv file
ID_dropped_df = pd.DataFrame({'ID_dropped': [107, 112, 120, 134]})
# locate the shuffled IDs that are dropped
ID_dropped_df['ID_shuffled_dropped'] = ID_df.loc[ID_df['ID'].isin(ID_dropped_df['ID_dropped'])]['ID_shuffled'].values

# save the dropped IDs dataframe to a csv file
ID_dropped_df.to_csv('Test_unshuffled/ID_dropped_df.csv', index=False)
