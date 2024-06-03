import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
# supress warnings
import warnings
warnings.filterwarnings("ignore")

# load training data
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

y = df_train['label']
X = df_train.drop(['label'],axis=1)

lr = LogisticRegression()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
svc = SVC(kernel='linear')
mlp = MLPClassifier()

# Define the parameter grid for each classifier you want to tune
rf_param_grid = {'n_estimators': [300, 500, 1000],
                 'max_depth': [None, 5],
                 'min_samples_split': [5, 10]}

# Define the parameter grid for logistic regression
lr_param_grid = {'C': np.logspace(-3,3,10),
                 'solver': ['newton-cg', 'lbfgs'],
                 'max_iter': [200, 1000, 5000]
                 }

gb_param_grid = {'learning_rate': [0.1, 0.01, 0.01],
                 'max_depth': [None, 5],
                 'min_samples_split': [5, 10],
                 'n_estimators': [100, 1000]
                 }

svc_param_grid = {'C': [0.01, 0.1, 1, 10, 100],
                  'gamma': ['scale', 'auto'],
                  'probability': [True]}

mlp_param_grid = {"alpha":[1e-2, 1e-1, 1],
                  "hidden_layer_sizes":[(30,),(30, 30),(30, 30, 30)],
                  "max_iter":[2000, 5000]
                  }

# create a dictionary of fitted_models and their respective parameter grids
models_and_parameters = [
    (lr, lr_param_grid),
    (rf, rf_param_grid),
    (gb, gb_param_grid),
    (svc, svc_param_grid),
    (mlp, mlp_param_grid),
]
# # create a dictionary of fitted_models and their respective parameter grids
# models_and_parameters = [
#     (lr, lr_param_grid)
# ]
# # Define a list to store the results
# results = []
#
# # Loop over a range of random seeds
# for seed in range(0, 50):
#     kf = KFold(n_splits=5, shuffle=True, random_state=seed)
#
#     best_score = 0
#     best_estimator = None
#     best_val_score = 0
#     best_features = None
#
#     selector = SelectKBest(score_func=f_classif, k=5)
#     X_selected_features = selector.fit_transform(X, y)
#
#     for model, params in models_and_parameters:
#         grid = GridSearchCV(estimator=model, param_grid=params, cv=kf, scoring='accuracy')
#         grid.fit(X_selected_features, y)
#         score = grid.best_score_
#
#         val_scores = cross_val_score(grid.best_estimator_, X_selected_features, y, cv=kf)
#         mean_val_score = np.mean(val_scores)
#
#         val_auc_scores = cross_val_score(grid.best_estimator_, X_selected_features, y, cv=kf,
#                                          scoring=make_scorer(roc_auc_score))
#         mean_val_auc = np.mean(val_auc_scores)
#
#         if mean_val_score > best_score:
#             best_score = score
#             best_estimator = grid.best_estimator_
#             best_val_score = mean_val_score
#             best_features = X.columns[selector.get_support()]
#
#     results.append({
#         'Seed': seed,
#         'Best Score': best_score,
#         'Best Estimator': best_estimator,
#         'Best Validation Score': best_val_score,
#         'Best Features': best_features
#     })
#
# # Find the best result
# best_result = max(results, key=lambda x: x['Best Validation Score'])
#
# print(f"The best random seed is {best_result['Seed']}, which yields a best validation AUC score of {best_result['Best Validation Score']}")


# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=4)

best_score = 0
best_estimator = None
best_val_score = 0
best_features = None

# Apply SelectKBest for feature selection
selector = SelectKBest(score_func=f_classif, k=5)  # You can change k to your preferred number
X_selected_features = selector.fit_transform(X, y)

# Initialize dataframe to store scores
score_df = pd.DataFrame(columns=['Model', 'Best Params', 'Avg Validation Accuracy', 'Avg Validation AUC'])

for model, params in models_and_parameters:

    # Grid search for hyperparameter tuning
    grid = GridSearchCV(estimator=model, param_grid=params, cv=kf, scoring=make_scorer(roc_auc_score))
    grid.fit(X_selected_features, y)
    score = grid.best_score_

    # accuracy score
    val_scores = cross_val_score(grid.best_estimator_, X_selected_features, y, cv=kf)
    mean_val_score = np.mean(val_scores)
    # auc score
    val_auc_scores = cross_val_score(grid.best_estimator_, X_selected_features, y, cv=kf,
                                     scoring=make_scorer(roc_auc_score))
    mean_val_auc = np.mean(val_auc_scores)

    if mean_val_score > best_score:
        best_score = score
        best_estimator = grid.best_estimator_
        best_val_score = mean_val_score
        best_features = X.columns[selector.get_support()]  # Gets the best features


    score_df = pd.concat([score_df, pd.DataFrame([{
        'Model': type(model).__name__,
        'Best Params': grid.best_params_,
        'Avg Validation Accuracy': mean_val_score,
        'Avg Validation AUC': mean_val_auc,
    }])], ignore_index=True)


# save score_df to csv
score_df.to_csv('fitted_models/full_model_score_df.csv', index=False)
print('Best training score:', best_score)
print('Best validation score:', best_val_score)
print('Best estimator:', best_estimator)
print('Best features:', best_features)

# save best estimator
joblib.dump(best_estimator, 'fitted_models/full_parameters_best_estimator.pkl')

# load test data
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

# choose the features from best_features
df_test = df_test[best_features]


# make predictions
results_test_all_variables = best_estimator.predict(df_test)
results_test_proba_all_variables = best_estimator.predict_proba(df_test)


df_test['label'] = results_test_all_variables

# save the results
df_test.to_csv('Test_shuffled/df_test_predicted_all_variables.csv')

results_test_proba_all_variables = pd.DataFrame(results_test_proba_all_variables, columns=['low', 'high'])
results_test_proba_all_variables['ID'] = df_test.index
results_test_proba_all_variables.to_csv('Test_shuffled/results_test_proba_all_variables.csv', index=False)
