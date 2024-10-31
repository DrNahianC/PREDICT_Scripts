import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.metrics import auc
from matplotlib import pyplot as plt

# load predicted test labels
df_test = pd.read_csv('Test_shuffled/PAF_CME_df_test_predicted.csv')
# read test labels
df_GT = pd.read_csv('Test_shuffled/df_ID_LGM_test.csv')
# rename ID to shuffled_ID
df_GT = df_GT.rename(columns={'ID':'ID_shuffled'})
# read mapper
df_mapper = pd.read_csv('Test_unshuffled/ID_df.csv')
# merge mapper and df_GT
df_GT = pd.merge(df_mapper, df_GT, on='ID_shuffled')
# merge df_GT and df_test
df_test = df_test.reset_index()
df_test = df_test.rename(columns={'id':'ID'})
df_final_test = pd.merge(df_GT, df_test, on='ID')

# calculate the number of matching elements
matches = np.sum(df_final_test['class'] == df_final_test['label'])

# calculate the accuracy
accuracy = matches / len(df_final_test['class'])
print('Accuracy:', accuracy)

# load test probabilities
df_test_proba = pd.read_csv('Test_shuffled/PAF_CME_results_test_proba.csv')
# merge df_test_proba and df_final_test
df_final_test_PAF_CME = pd.merge(df_final_test, df_test_proba, on='ID')

# plot the auc-roc curve and display auc on the legend
from sklearn.metrics import roc_curve
lr_fpr_test, lr_tpr_test, lr_thresholds_test = roc_curve(df_final_test_PAF_CME['class'], df_final_test_PAF_CME['high'])
# plot the ROC curve for the voting classifier, including auc in the legend, and accuracy in the title
plt.figure(figsize=(10, 8))
plt.plot(lr_fpr_test, lr_tpr_test, label='AUC = {:.2f}'.format(auc(lr_fpr_test, lr_tpr_test)), linewidth=4, color = "black")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.legend(loc='best', fontsize=20)
plt.show()
plt.savefig('figures/PAF_CME_roc_test.png')
# save figures
print('AUC:', auc(lr_fpr_test, lr_tpr_test))


# plot the pain diary data based on the predicted labels
# load the diary
df_yawn = pd.read_csv('Test_shuffled/df_yawn_filled_test.csv', index_col='ID')
df_chew = pd.read_csv('Test_shuffled/df_chew_filled_test.csv', index_col='ID')
df_diary = df_yawn + df_chew
# high_pain_severity
ID_high = df_final_test.loc[df_final_test['label'] == 1, 'ID_shuffled'].to_list()
df_high_pain_severity = df_diary.loc[df_diary.index.isin(ID_high), :]
ID_low = df_final_test.loc[df_final_test['label'] == 0, 'ID_shuffled'].to_list()
df_low_pain_severity = df_diary.loc[df_diary.index.isin(ID_low), :]
# plot the diary according to the df_test.label
plt.figure(figsize=(26, 27))
# mean values of the diary
plt.plot(df_high_pain_severity.mean(), label='Predicted high pain', linewidth=3, color="red")
plt.plot(df_low_pain_severity.mean(), label='Predicted low pain', linewidth=3, color="blue")
# add confidence interval
plt.fill_between(df_high_pain_severity.columns, df_high_pain_severity.mean() - df_high_pain_severity.std(),
                 df_high_pain_severity.mean() + df_high_pain_severity.std(), color="red", alpha=0.2)
plt.fill_between(df_low_pain_severity.columns, df_low_pain_severity.mean() - df_low_pain_severity.std(),
                 df_low_pain_severity.mean() + df_low_pain_severity.std(), color="blue", alpha=0.2)
# vertical xlabels
#plt.xlabel('Timepoint', fontsize=48)
plt.xticks([0, 2, 4, 6, 8, 59], ['Day1AM', 'Day2AM','Day3AM', 'Day4AM','Day5AM','Day30PM'], rotation=90, fontsize = 48)
plt.yticks(fontsize=48)
plt.ylim([0,12])
plt.ylabel('Pain Score (chew + yawn)', fontsize=48)
plt.legend(loc='best', fontsize=48)
plt.grid(False)
plt.savefig('figures/PAF_CME_pain_severity_test.png')
plt.show()
plt.close('all')

