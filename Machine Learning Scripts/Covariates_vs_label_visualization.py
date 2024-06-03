import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import statsmodels.api as sm
# supress warnings
import warnings
warnings.filterwarnings("ignore")


# load data
labels_train = 'Train/df_ID_train_LGM.csv'
df_labels_train = pd.read_csv(labels_train)
Sensorimotor_PAF_train = 'Train/Sensorimotor_Peak_Alpha_Frequency_train.xlsx'
df_PAF_train = pd.read_excel(Sensorimotor_PAF_train)
# select ids that are in labels_train
df_PAF_train = df_PAF_train.loc[df_PAF_train.id.isin(df_labels_train.low.to_list() + df_labels_train.high.to_list())]
Map_Volume_train = 'Train/Map_Volume_train.xlsx'
df_Map_Volume_train = pd.read_excel(Map_Volume_train)
# select ids that are in labels_train
df_Map_Volume_train = df_Map_Volume_train.loc[df_Map_Volume_train.id.isin(df_labels_train.low.to_list() + df_labels_train.high.to_list())]
imp = IterativeImputer(max_iter=100, random_state=0)
df_Map_Volume_train[['Volume_Day0','Volume_Day2','Volume_Day5']] = imp.fit_transform(df_Map_Volume_train[['Volume_Day0','Volume_Day2','Volume_Day5']])
df_Map_Volume_train['CME'] = df_Map_Volume_train.Volume_Day5 - df_Map_Volume_train.Volume_Day0
MapVol = df_Map_Volume_train[['id','CME']]
MapVol.loc[MapVol['CME']>=0,'CME'] = 1
MapVol.loc[MapVol['CME']<0,'CME'] = 0

df_train = pd.merge(df_PAF_train, MapVol, on='id')
df_train['label'] = 0
df_train.loc[df_train.id.isin(df_labels_train['high']),'label'] = 1

# load other variables
demographics_file = 'Train/Demographics_train.xlsx'
covariate_file = 'Train/Covariate_Data_train.xlsx'

df_demographics_train = pd.read_excel(demographics_file)
# rename record_id to id
df_demographics_train = df_demographics_train.rename(columns={'record_id':'id'})
df_covariates_train = pd.read_excel(covariate_file)
# rename record_id to id
df_covariates_train = df_covariates_train.rename(columns={'record_id':'id'})

# select the features from demographics and covariates dataframes
df_train = pd.merge(df_train, df_demographics_train, on='id')
df_train = pd.merge(df_train, df_covariates_train, on='id')
# label female and male as 2 and 1
df_train.loc[df_train.Sex == 'Male','Sex'] = 1
df_train.loc[df_train.Sex == 'Female','Sex'] = 2
df_train = df_train.set_index('id')
df_train = df_train.astype(float)

# Now load the test data
to_drop = [107, 112, 120, 134]

MAP_test = pd.read_excel('Test_shuffled/Map_Volume_test_unshuffled.xlsx', index_col='id')
imp = IterativeImputer(max_iter=10, random_state=0)
MAP_test[['Volume_Day0','Volume_Day2','Volume_Day5']] = imp.fit_transform(MAP_test[['Volume_Day0','Volume_Day2','Volume_Day5']])
MAP_test['CME'] = MAP_test.Volume_Day5 - MAP_test.Volume_Day0
MAP_test.loc[MAP_test['CME']>=0,'CME'] = 1
MAP_test.loc[MAP_test['CME']<0,'CME'] = 0
MAP_test = MAP_test.loc[~MAP_test.index.isin(to_drop),'CME']

# load sensorimotor PAF and select ids
sensorimotor_PAF_test = pd.read_excel('Test_shuffled/Sensorimotor_Peak_Alpha_Frequency_test_unshuffled.xlsx', index_col='id')
sensorimotor_PAF_test = sensorimotor_PAF_test.loc[~sensorimotor_PAF_test.index.isin(to_drop),'sensorimotor_paf']

# merge the dataframes
df_test = pd.concat([sensorimotor_PAF_test, MAP_test], axis=1)
#
# load test demographic and covariates data
Demographic_test = pd.read_excel('Test_shuffled/Demographics_test_unshuffled.xlsx', index_col='id')
Demographic_test = Demographic_test.loc[~Demographic_test.index.isin(to_drop),:]
Covariates_test = pd.read_excel('Test_shuffled/Covariate_Data_test_unshuffled.xlsx')
# rename record_id to id
Covariates_test = Covariates_test.rename(columns={'record_id': 'id'})
# set id as index
Covariates_test = Covariates_test.set_index('id')
Covariates_test = Covariates_test.loc[~Covariates_test.index.isin(to_drop),:]

# select the features from demographics and covariates dataframes
df_test = pd.merge(df_test, Demographic_test, on='id')
df_test = pd.merge(df_test, Covariates_test, on='id')
# label female and male as 2 and 1
df_test.loc[df_test.Sex == 'Male','Sex'] = 1
df_test.loc[df_test.Sex == 'Female','Sex'] = 2

df_test.fillna(df_test.mean(), inplace=True)


# load predicted test labels
df_test_all = pd.read_csv('Test_shuffled/df_test_predicted_all_variables.csv')

# read test labels
df_GT = pd.read_csv('Test_shuffled/df_ID_LGM_test.csv')
# rename ID to shuffled_ID
df_GT = df_GT.rename(columns={'ID':'ID_shuffled'})
# read mapper
df_mapper = pd.read_csv('Test_unshuffled/ID_df.csv')
# merge mapper and df_GT
df_GT = pd.merge(df_mapper, df_GT, on='ID_shuffled')
# merge df_GT and df_test_all
df_test_all = df_test_all.reset_index()
df_test_all = df_test_all.rename(columns={'id':'ID'})
df_final_test_all = pd.merge(df_GT, df_test_all, on='ID')
# rename ID to id
df_final_test_all = df_final_test_all.rename(columns={'ID':'id'})
# set id as index
df_final_test_all = df_final_test_all.set_index('id')
# merge df_final_test_all and df_test
df_final_test_all = pd.merge(df_final_test_all['class'], df_test, on='id')
# rename class to label
df_final_test_all = df_final_test_all.rename(columns={'class':'label'})

# concatenate df_train and df_final_test_all
df = pd.concat([df_train, df_final_test_all], axis=0)

# correlation matrix
corr_train = df_train.corr().label
corr_test = df_final_test_all.corr().label


# get the number of labels for different genders
male_high_train = df_train.loc[(df_train.Sex == 1) & (df_train.label == 1), 'Age'].count()
male_low_train = df_train.loc[(df_train.Sex == 1) & (df_train.label == 0), 'Age'].count()
female_high_train = df_train.loc[(df_train.Sex == 2) & (df_train.label == 1), 'Age'].count()
female_low_train = df_train.loc[(df_train.Sex == 2) & (df_train.label == 0), 'Age'].count()

# get the number of labels for different genders
CME_neg_high_train = df_train.loc[(df_train.CME == 0) & (df_train.label == 1), 'Age'].count()
CME_pos_high_train = df_train.loc[(df_train.CME == 1) & (df_train.label == 1), 'Age'].count()
CME_neg_low_train = df_train.loc[(df_train.CME == 0) & (df_train.label == 0), 'Age'].count()
CME_pos_low_train = df_train.loc[(df_train.Sex == 1) & (df_train.label == 0), 'Age'].count()

# calculate odd ratio between CME and label
# Create a logistic regression model
logit_model_CME = sm.Logit(np.array(df_train.label.to_list()), sm.add_constant(np.array(df_train.CME.to_list())))
# Fit the model
result_CME = logit_model_CME.fit()
odds_CME = np.exp(result_CME.params)[1]


# odds ratio for sex
logit_model_Sex = sm.Logit(np.array(df_train.label.to_list()), sm.add_constant(np.array(df_train.Sex.to_list())))
# Fit the model
result_Sex = logit_model_Sex.fit()
odds_Sex = np.exp(result_Sex.params)[1]


# rename the labels to High and Low
df_train['label'] = df_train['label'].replace({1: 'High', 0: 'Low'})
Label = ['Low','High']
male_train = [male_low_train,male_high_train]
female_train = [female_low_train,female_high_train]

CME_neg_train = [CME_neg_low_train, CME_neg_high_train]
CME_pos_train = [CME_pos_low_train, CME_pos_high_train]

# create barplots between label and 'sensorimotor_paf',
sns.set(style="whitegrid")
sns.set(font_scale=1.5)
#fig, ax = plt.subplots(1, 3, figsize=(20, 5))
colors = ['blue', 'red']
fig, ax = plt.subplots()
sns.boxplot(x='label', y='sensorimotor_paf', data=df_train, order=['Low','High'], palette=colors)
ax.set_ylabel('Sensorimotor PAF')
ax.set_xlabel('Pain Label')
ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/boxplots_train_paf.png', dpi=300, bbox_inches='tight')

# Create a stacked bar chart
fig, ax = plt.subplots()
ax.bar('Low', CME_neg_low_train, bottom=CME_pos_low_train, label='CME: Depressor', color="blue", alpha = 0.2)
ax.bar('Low', CME_pos_low_train, label='CME: Facilitator', color="blue", alpha = 0.8)
ax.bar('High', CME_neg_high_train, bottom=CME_pos_high_train, label='CME: Depressor', color="red", alpha = 0.2)
ax.bar('High', CME_pos_high_train, label='CME: Facilitator', color="red", alpha = 0.8)
ax.set_xlabel('Pain Label')
ax.set_ylabel('Facilitator:Depressor Split')
# put odds_Sex as title
ax.set_title('Odds ratio: {:.2f}'.format(odds_CME))
#ax.legend()
ax.grid(False)
ax.set_facecolor('white')

plt.show()
# save the figure
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/stacked_boxplots_train_CME.png', dpi=300, bbox_inches='tight')

## Plot Covariates 
#PCS Helplessness
sns.set(style="whitegrid")
sns.set(font_scale=1.5)
fig, ax = plt.subplots()
sns.boxplot(x='label', y='pcs_helplessness', data=df_train, order=['Low','High'], palette=colors)
ax.set_ylabel('PCS Helplessness')
ax.set_xlabel('Pain Label')
ax.set_facecolor('white')
ax.grid(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/boxplots_train_pcs_helplessness.png', dpi=300, bbox_inches='tight')
plt.show()

#PCS Total
sns.set(style="whitegrid")
sns.set(font_scale=1.5)
fig, ax = plt.subplots()
sns.boxplot(x='label', y='pcs_total', data=df_train, order=['Low','High'], palette=colors)
ax.set_ylabel('PCS Total')
ax.set_xlabel('Pain Label')
ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/boxplots_train_pcs_total.png', dpi=300, bbox_inches='tight')    
plt.show()
# save the figure


# Create a stacked bar chart
fig, ax = plt.subplots()
ax.bar('Low', male_low_train, bottom=female_low_train, label='Male', color="blue", alpha = 0.2)
ax.bar('Low', female_low_train, label='Female', color="blue", alpha = 0.8)
ax.bar('High', male_high_train, bottom=female_high_train, label='Male', color="red", alpha = 0.2)
ax.bar('High', female_high_train, label='Female', color="red", alpha = 0.8)
ax.set_xlabel('Pain Label')
ax.set_ylabel('Female:Male Split')
# put odds_Sex as title
ax.set_title('Odds ratio: {:.2f}'.format(odds_Sex))
#ax.legend()
ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
plt.show()
# save the figure
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/stacked_boxplots_train_sex.png', dpi=300, bbox_inches='tight')





# get the number of labels for different genders
male_high_test = df_final_test_all.loc[(df_final_test_all.Sex == 1) & (df_final_test_all.label == 1), 'Age'].count()
male_low_test = df_final_test_all.loc[(df_final_test_all.Sex == 1) & (df_final_test_all.label == 0), 'Age'].count()
female_high_test = df_final_test_all.loc[(df_final_test_all.Sex == 2) & (df_final_test_all.label == 1), 'Age'].count()
female_low_test = df_final_test_all.loc[(df_final_test_all.Sex == 2) & (df_final_test_all.label == 0), 'Age'].count()

# get the number of labels for different genders
CME_neg_high_test = df_final_test_all.loc[(df_final_test_all.CME == 0) & (df_final_test_all.label == 1), 'Age'].count()
CME_pos_high_test = df_final_test_all.loc[(df_final_test_all.CME == 1) & (df_final_test_all.label == 1), 'Age'].count()
CME_neg_low_test = df_final_test_all.loc[(df_final_test_all.CME == 0) & (df_final_test_all.label == 0), 'Age'].count()
CME_pos_low_test = df_final_test_all.loc[(df_final_test_all.Sex == 1) & (df_final_test_all.label == 0), 'Age'].count()

# calculate odd ratio between CME and label
# Create a logistic regression model
logit_model_CME = sm.Logit(np.array(df_final_test_all.label.to_list()), sm.add_constant(np.array(df_final_test_all.CME.to_list())))
# Fit the model
result_CME = logit_model_CME.fit()
odds_CME = np.exp(result_CME.params)[1]


# odds ratio for sex
logit_model_Sex = sm.Logit(np.array(df_final_test_all.label.to_list()), sm.add_constant(np.array(df_final_test_all.Sex.to_list())))
# Fit the model
result_Sex = logit_model_Sex.fit()
odds_Sex = np.exp(result_Sex.params)[1]


Label = ['Low','High']
male_test = [male_low_test,male_high_test]
female_test = [female_low_test,female_high_test]

CME_neg_test = [CME_neg_low_test, CME_neg_high_test]
CME_pos_test = [CME_pos_low_test, CME_pos_high_test]

# create boxplots between label and 'sensorimotor_paf', 'CME', 'pcs_helplessness', 'pcs_total',
# rename the labels to High and Low
# low to the left and high to the right

df_final_test_all['label'] = df_final_test_all['label'].replace({ 0: 'Low',1: 'High'})
sns.set(style="whitegrid")
sns.set(font_scale=1.5)
#fig, ax = plt.subplots(1, 3, figsize=(20, 5))
colors = ['blue', 'red']
fig, ax = plt.subplots()
sns.boxplot(x='label', y='sensorimotor_paf', data=df_final_test_all, order=['Low','High'], palette=colors)
ax.set_ylabel('Sensorimotor PAF')
ax.set_xlabel('Pain Label')
ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/boxplots_test_paf.png', dpi=300, bbox_inches='tight')



# Create a stacked bar chart
fig, ax = plt.subplots()
ax.bar('Low', CME_neg_low_test, bottom=CME_pos_low_test, label='CME: Depressor',color="blue", alpha = 0.2)
ax.bar('Low', CME_pos_low_test, label='CME: Facilitator',color="blue", alpha = 0.8)
ax.bar('High', CME_neg_high_test, bottom=CME_pos_high_test, label='CME: Depressor',color="red", alpha = 0.2)
ax.bar('High', CME_pos_high_test, label='CME: Facilitator',color="red", alpha = 0.8)
ax.set_xlabel('Pain Label')
ax.set_ylabel('Facilitator:Depressor Split')
# put odds_Sex as title
ax.set_title('Odds ratio: {:.2f}'.format(odds_CME))
ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
#ax.legend()
plt.show()
# save the figure
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/stacked_boxplots_test_CME.png', dpi=300, bbox_inches='tight')
plt.close('all')



# Create a stacked bar chart
fig, ax = plt.subplots()
ax.bar('Low', CME_neg_low_train, bottom=CME_pos_low_train, label='CME: Depressor', color="blue", alpha = 0.2)
ax.bar('Low', CME_pos_low_train, label='CME: Facilitator', color="blue", alpha = 0.8)
ax.bar('High', CME_neg_high_train, bottom=CME_pos_high_train, label='CME: Depressor', color="red", alpha = 0.2)
ax.bar('High', CME_pos_high_train, label='CME: Facilitator', color="red", alpha = 0.8)
ax.set_xlabel('Pain Label')
ax.set_ylabel('Facilitator:Depressor Split')
# put odds_Sex as title
ax.set_title('Odds ratio: {:.2f}'.format(odds_CME))
ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
#ax.legend()
plt.show()
# save the figure
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/stacked_boxplots_train_CME.png', dpi=300, bbox_inches='tight')


## Plot Covariates
#PCS Helplessness 
sns.set(style="whitegrid")
sns.set(font_scale=1.5)
fig, ax = plt.subplots()
sns.boxplot(x='label', y='pcs_helplessness', data=df_final_test_all, order=['Low','High'], palette=colors)
ax.set_ylabel('PCS Helplessness')
ax.set_xlabel('Pain Label')
ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/boxplots_test_pcs_helplessness.png', dpi=300, bbox_inches='tight')    
plt.show()

#PCS total
sns.set(style="whitegrid")
sns.set(font_scale=1.5)
fig, ax = plt.subplots()
sns.boxplot(x='label', y='pcs_total', data=df_final_test_all, order=['Low','High'], palette=colors)
ax.set_ylabel('PCS Total')
ax.set_xlabel('Pain Label')
ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/boxplots_test_pcs_total.png', dpi=300, bbox_inches='tight')    
plt.show()


# Create a stacked bar chart
fig, ax = plt.subplots()
ax.bar('Low', male_low_test, bottom=female_low_test, label='Male', color="blue", alpha = 0.2)
ax.bar('Low', female_low_test, label='Female', color="blue", alpha = 0.8)
ax.bar('High', male_high_test, bottom=female_high_test, label='Male', color="red", alpha = 0.2)
ax.bar('High', female_high_test, label='Female', color="red", alpha = 0.8)
ax.set_xlabel('Pain Label')
ax.set_ylabel('Female:Male Split')
# put odds_Sex as title
ax.set_title('Odds ratio: {:.2f}'.format(odds_Sex))
#ax.grid(False)
ax.set_facecolor('white')
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')
#ax.legend()
plt.show()
# save the figure
fig.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/stacked_boxplots_test_sex.png', dpi=300, bbox_inches='tight')

