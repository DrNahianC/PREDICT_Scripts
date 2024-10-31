install.packages("Hmisc")
install.packages("rms")
install.packages("Rmisc")
install.packages("nlme")
install.packages("readxl")
install.packages("reshape2")
install.packages("ggpubr")
install.packages("lcmm")


require(Hmisc)
require(ggplot2)
require(rms)
require(Rmisc)
require(nlme)
require(readxl)
require(reshape2)
require(ggpubr)
require(lcmm)


setwd("PREDICT_Scripts-main/Machine Learning Scripts") #change this to where your script directory is on your machine 
chew_pain_df=read.csv("Train/df_chew_filled_train.csv")
yawn_pain_df=read.csv("Train/df_yawn_filled_train.csv")
all_pain_df_train=yawn_pain_df + chew_pain_df
all_pain_df_train[c("ID")] = yawn_pain_df[c("ID")]
# take the first 100 to train
all_pain_df_long_train = melt(data = all_pain_df_train[, 1:14],
                              id.vars = c("ID"),
                              variable.name = "t",
                              value.name = "y")
lcga1 <-hlme(y ~ t, subject = "ID", ng = 1, data = all_pain_df_long_train, verbose = FALSE)
lcga2 <-gridsearch(rep = 8, maxiter = 100, minit = lcga1,hlme(y ~ t, subject = "ID",ng = 2, data = all_pain_df_long_train, mixture = ~ t, verbose = FALSE))
class_1_ID = lcga2$pprob[order(lcga2$pprob$prob1)[1:40],c(1)]
class_2_ID = lcga2$pprob[order(lcga2$pprob$prob1)[61:100],c(1)]
classA = all_pain_df_train[all_pain_df_train$ID %in% class_1_ID,]
classB = all_pain_df_train[all_pain_df_train$ID %in% class_2_ID,]
if (mean(apply(classA[, c(2:11)], 2, mean)) < mean(apply(classB[, c(2:11)], 2, mean))) {
  # if subjects in classA has lower average pain, then prob1 corresponds to the high pain severity class
  ID_low_LGM = class_1_ID
  ID_high_LGM = class_2_ID
  # locate the probability threshold for low and high pain severity, in terms of the high probability
  low_threshold = lcga2$pprob[order(lcga2$pprob$prob1)[41],c(3)]
  high_threshold = lcga2$pprob[order(lcga2$pprob$prob1)[60],c(3)]
}else {
  # if subjects in classA has higher average pain, then prob1 corresponds to the low pain severity class
  ID_low_LGM = class_2_ID
  ID_high_LGM = class_1_ID
  # locate the probability threshold for low and high pain severity, in terms of the high probability
  low_threshold = lcga2$pprob[order(lcga2$pprob$prob2)[41],c(4)]
  high_threshold = lcga2$pprob[order(lcga2$pprob$prob2)[60],c(4)]
}
# save
df_ID_LGM_train = data.frame(low = ID_low_LGM, high = ID_high_LGM)
write.csv(df_ID_LGM_train, "Train\\df_ID_train_LGM.csv", row.names=FALSE)
### The latent growth model is lock to predict classes for the test set, where


### all subjects' IDs are shuffled in the test set
chew_pain_df_test=read.csv("Test_shuffled/df_chew_filled_test.csv")
yawn_pain_df_test=read.csv("Test_shuffled/df_yawn_filled_test.csv")
all_pain_df_test=chew_pain_df_test + yawn_pain_df_test
all_pain_df_test[c("ID")] = yawn_pain_df_test[c("ID")]
# rest 50 to test
all_pain_df_long_test = melt(data = all_pain_df_test[, 1:14],
                             id.vars = c("ID"),
                             variable.name = "t",
                             value.name = "y")
# prediction
predicted_values <- predictClass(lcga2, newdata = all_pain_df_long_test)
class_1_ID_test = predicted_values[order(predicted_values$prob1)[1:20],c(1)]
class_2_ID_test = predicted_values[order(predicted_values$prob1)[31:50],c(1)]
classA_test = all_pain_df_test[all_pain_df_test$ID %in% class_1_ID_test,]
classB_test = all_pain_df_test[all_pain_df_test$ID %in% class_2_ID_test,]
if (mean(apply(classA_test[, c(2:11)], 2, mean)) < mean(apply(classB_test[, c(2:11)], 2, mean))) {
  # true if classA (1) is low, classB (2) is high
  # prob1 is the probability corresponds to high pain severity
  ID_low_LGM_test = predicted_values$ID[predicted_values$prob1<low_threshold]
  ID_high_LGM_test = predicted_values$ID[predicted_values$prob1>high_threshold]
}else{
  # false if classB (2) is low, classA (1) is high
  # prob2 is the probability corresponds to high pain severity
  ID_low_LGM_test = predicted_values$ID[predicted_values$prob2<low_threshold]
  ID_high_LGM_test = predicted_values$ID[predicted_values$prob2>high_threshold]
}
# Create a vector with the class labels
class_labels <- c(rep(1, length(ID_high_LGM_test)),
                  rep(0, length(ID_low_LGM_test)))
# Create a vector with all the IDs
test_IDs <- c(ID_high_LGM_test, ID_low_LGM_test)
# Combine the ID and class vectors into a dataframe
df_ID_LGM_test <- data.frame(ID = test_IDs, class = class_labels)
write.csv(df_ID_LGM_test, "Test_shuffled\\df_ID_LGM_test.csv", row.names=FALSE)