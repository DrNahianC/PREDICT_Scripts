import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

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

# calculate the number of matching elements
matches_all = np.sum(df_final_test_all['class'] == df_final_test_all['label'])

# calculate the accuracy
accuracy_all = matches_all / len(df_final_test_all['class'])

print('Accuracy:', accuracy_all)

# load test probabilities
df_test_proba_all = pd.read_csv('Test_shuffled/results_test_proba_all_variables.csv')
# merge df_test_proba and df_final_test
df_final_test_all = pd.merge(df_final_test_all, df_test_proba_all, on='ID')

# plot the auc-roc curve and display auc on the legend
rf_fpr_test_all, rf_tpr_test_all, rf_thresholds_test_all = roc_curve(df_final_test_all['class'], df_final_test_all['high'])
# plot the ROC curve for the voting classifier, including auc in the legend, and accuracy in the title
plt.figure(figsize=(10, 8))
plt.plot(rf_fpr_test_all, rf_tpr_test_all, label='AUC = {:.2f}'.format(auc(rf_fpr_test_all, rf_tpr_test_all)), linewidth=4, color = "black")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(loc='best', fontsize=20)
plt.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/roc_test_all.png')
plt.show()
# save figures


# plot the pain diary data based on the predicted labels
# load the diary
df_yawn = pd.read_csv('Test_shuffled/df_yawn_filled_test.csv', index_col='ID')
df_chew = pd.read_csv('Test_shuffled/df_chew_filled_test.csv', index_col='ID')
df_diary = df_yawn + df_chew
# high_pain_severity
ID_high_all = df_final_test_all.loc[df_final_test_all['label'] == 1, 'ID_shuffled'].to_list()
df_high_pain_severity_all = df_diary.loc[df_diary.index.isin(ID_high_all), :]
ID_low_all = df_final_test_all.loc[df_final_test_all['label'] == 0, 'ID_shuffled'].to_list()
df_low_pain_severity_all = df_diary.loc[df_diary.index.isin(ID_low_all), :]
# plot the diary according to the df_test.label
plt.figure(figsize=(26, 27))
# mean values of the diary
plt.plot(df_high_pain_severity_all.mean(), label='Predicted high pain severity', linewidth=3, color = "red")
plt.plot(df_low_pain_severity_all.mean(), label='Predicted low pain severity', linewidth=3, color = "blue")
# add confidence interval
plt.fill_between(df_high_pain_severity_all.columns, df_high_pain_severity_all.mean() - df_high_pain_severity_all.std(),
                 df_high_pain_severity_all.mean() + df_high_pain_severity_all.std(), color = "red", alpha=0.2)
plt.fill_between(df_low_pain_severity_all.columns, df_low_pain_severity_all.mean() - df_low_pain_severity_all.std(),
                 df_low_pain_severity_all.mean() + df_low_pain_severity_all.std(), color = "blue", alpha=0.2)
# vertical xlabels
#plt.xlabel('Timepoint', fontsize=48)
plt.xticks([0, 2, 4, 6, 8, 59], ['Day1AM', 'Day2AM','Day3AM', 'Day4AM','Day5AM','Day30PM'], rotation=90, fontsize = 48)
plt.yticks(fontsize=48)
plt.ylim([0,12])
plt.ylabel('Pain Score (chew + yawn)', fontsize=48)
plt.legend(loc='best', fontsize=48)
plt.grid(False)
plt.savefig('D:/PREDICT - Projects/Main Outcomes Paper/Chuan Report/figures/pain_severity_test_all.png')
plt.show()
plt.close('all')