import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

import warnings
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# Define file paths
Pain_Diary_Data_train = 'Train/Pain_Diary_Data_train.xlsx'

# Read in data
PD_chewAM_df = pd.read_excel(Pain_Diary_Data_train, sheet_name='Pain_Chew_AM')
PD_chewPM_df = pd.read_excel(Pain_Diary_Data_train, sheet_name='Pain_Chew_PM')
PD_yawnAM_df = pd.read_excel(Pain_Diary_Data_train, sheet_name='Pain_Yawn_AM')
PD_yawnPM_df = pd.read_excel(Pain_Diary_Data_train, sheet_name='Pain_Yawn_PM')

# Drop subjects who dropped out
to_drop = [13, 23, 47, 75, 77]

PD_chewAM_df = PD_chewAM_df.loc[~PD_chewAM_df.ID.isin(to_drop)]
PD_chewAM_df.set_index('ID',inplace=True)
PD_chewPM_df = PD_chewPM_df.loc[~PD_chewPM_df.ID.isin(to_drop)]
PD_chewPM_df.set_index('ID',inplace=True)
PD_yawnAM_df = PD_yawnAM_df.loc[~PD_yawnAM_df.ID.isin(to_drop)]
PD_yawnAM_df.set_index('ID',inplace=True)
PD_yawnPM_df = PD_yawnPM_df.loc[~PD_yawnPM_df.ID.isin(to_drop)]
PD_yawnPM_df.set_index('ID',inplace=True)

# Merge AM and PM data
PD_chewAM_df.columns = PD_chewAM_df.columns.str.replace("Day_", "")
PD_chewPM_df.columns = PD_chewPM_df.columns.str.replace("Day_", "")
PD_yawnAM_df.columns = PD_yawnAM_df.columns.str.replace("Day_", "")
PD_yawnPM_df.columns = PD_yawnPM_df.columns.str.replace("Day_", "")

PD_chewAM_df = PD_chewAM_df.add_suffix('_AM')
PD_chewPM_df = PD_chewPM_df.add_suffix('_PM')
PD_yawnAM_df = PD_yawnAM_df.add_suffix('_AM')
PD_yawnPM_df = PD_yawnPM_df.add_suffix('_PM')

df_chew = pd.merge(PD_chewAM_df, PD_chewPM_df, left_index=True, right_index=True)
df_yawn = pd.merge(PD_yawnAM_df, PD_yawnPM_df, left_index=True, right_index=True)

df_chew = df_chew.reindex(sorted(df_chew.columns, key=lambda x: float(x[:-3])), axis=1)
df_yawn = df_yawn.reindex(sorted(df_yawn.columns, key=lambda x: float(x[:-3])), axis=1)

# Impute missing values using IterativeImputer
df_chew = df_chew.interpolate(axis=1)
df_chew_filled = df_chew.fillna(method='bfill', axis=1)
df_yawn= df_yawn.interpolate(axis=1)
df_yawn_filled = df_yawn.fillna(method='bfill', axis=1)
df_chew_filled.to_csv('Train/df_chew_filled_train.csv')
df_yawn_filled.to_csv('Train/df_yawn_filled_train.csv')