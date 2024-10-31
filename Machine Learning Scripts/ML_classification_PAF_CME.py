import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
import joblib
import numpy as np
from parameter_tuning import ClassifierTuner
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

df_train = df_train.drop(['id'],axis=1)
y = df_train['label']
X = df_train.drop(['label'],axis=1)

lr = LogisticRegression()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
svc = SVC(kernel='linear')
mlp = MLPClassifier()

# Define the parameter grid for each classifier you want to tune
rf_param_grid = {'n_estimators': [300, 500, 1000],
                 'max_depth': [None, 5, 10],
                 'min_samples_split': [2, 5, 10],
                 'bootstrap': [True, False]}

# Define the parameter grid for logistic regression
lr_param_grid = {'C': np.logspace(-3,3,30),
                 'solver': ['newton-cg', 'lbfgs'],
                 'max_iter': [200, 400, 2000, 5000]
                 }

gb_param_grid = {'learning_rate': [1, 0.1, 0.01, 0.01, 0.001],
                 'max_depth': [None, 2, 5],
                 'min_samples_split': [2, 5, 10],
                 'n_estimators': [300, 500, 1000]
                 }

svc_param_grid = {'C': [0.01, 0.1, 1, 10, 100],
                  'gamma': ['scale', 'auto']}

mlp_param_grid = {"alpha":[1e-3, 1e-2, 1e-1, 0, 1,10],
                  "hidden_layer_sizes":[(100,),(100,100),(100,100,100)],
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

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=34)

best_score = 0
best_estimator = None
best_val_score = 0
best_features = None

# Initialize dataframe to store scores
score_df = pd.DataFrame(columns=['Model', 'Best Params', 'Avg Validation Accuracy', 'Avg Validation AUC'])

for model, params in models_and_parameters:

    # Grid search for hyperparameter tuning
    grid = GridSearchCV(estimator=model, param_grid=params, cv=kf, scoring='accuracy', n_jobs=-1)
    grid.fit(X, y)
    score = grid.best_score_

    # accuracy score
    val_scores = cross_val_score(grid.best_estimator_, X, y, cv=kf)
    mean_val_score = np.mean(val_scores)
    # auc score
    val_auc_scores = cross_val_score(grid.best_estimator_, X, y, cv=kf,
                                     scoring=make_scorer(roc_auc_score))
    print(val_auc_scores)
    mean_val_auc = np.mean(val_auc_scores)

    if mean_val_score > best_score:
        best_score = score
        best_estimator = grid.best_estimator_
        best_val_score = mean_val_score


    score_df = pd.concat([score_df, pd.DataFrame([{
        'Model': type(model).__name__,
        'Best Params': grid.best_params_,
        'Avg Validation Accuracy': mean_val_score,
        'Avg Validation AUC': mean_val_auc,
    }])], ignore_index=True)


# save score_df to csv
score_df.to_csv('fitted_models/PAF_CME_model_score_df.csv', index=False)
# display the results
pd.set_option('display.max_columns', None)
print(score_df)

# save the model to fitted_model folder
joblib.dump(best_estimator, 'fitted_models/PAF_CME_best_estimator.pkl')

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
# standardize the data
# df_test_mtx = scaler.transform(df_test)

# make predictions
results_test = best_estimator.predict(df_test)
results_test_proba = best_estimator.predict_proba(df_test)


df_test['label'] = results_test

# save the results
df_test.to_csv('Test_shuffled/PAF_CME_df_test_predicted.csv')

results_test_proba = pd.DataFrame(results_test_proba, columns=['low', 'high'])
results_test_proba['ID'] = df_test.index
results_test_proba.to_csv('Test_shuffled/PAF_CME_results_test_proba.csv', index=False)


########################################################################################################################
# train_test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=23)


#Training AUC
lr_train_results_proba = best_estimator.predict_proba(X_train)
lr_fpr_train, lr_tpr_train, lr_thresholds_train = roc_curve(y_train, lr_train_results_proba[:,1])
auc(lr_fpr_train, lr_tpr_train)
print(auc(lr_fpr_train, lr_tpr_train))


# make predictions
results_val = best_estimator.predict(X_val)
results_val_proba = best_estimator.predict_proba(X_val)
lr_fpr_val, lr_tpr_val, lr_thresholds_val = roc_curve(y_val, results_val_proba[:,1])
print(auc(lr_fpr_val, lr_tpr_val))

matches_all = np.sum(results_val == y_val)

# calculate the accuracy
accuracy_all = matches_all / len(y_val)
print(accuracy_all)


# plot the ROC curve for the voting classifier
plt.figure(figsize=(10,8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(lr_fpr_val, lr_tpr_val, label='AUC: {:.3f}'.format(auc(lr_fpr_val, lr_tpr_val)), linewidth=4, color = "black")
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(loc='best', fontsize=20)
# save figure
plt.savefig('figures/Training_ROC_curve.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close('all')
