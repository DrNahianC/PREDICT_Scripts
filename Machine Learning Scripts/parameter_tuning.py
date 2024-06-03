from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

class ClassifierTuner:
    def __init__(self, X_train, y_train, random_state=30, cv=5):
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.cv = cv

    def tune_random_forest(self, param_grid):
        rf = RandomForestClassifier()
        clf_rf = RandomizedSearchCV(rf, param_grid, cv=self.cv, n_iter=10, n_jobs=-1, random_state=self.random_state, scoring='roc_auc')
        clf_rf.fit(self.X_train, self.y_train)
        return clf_rf.best_estimator_

    def tune_logistic_regression(self, param_grid):
        lr = LogisticRegression()
        clf_lr = RandomizedSearchCV(lr, param_grid, cv=self.cv, n_iter=10, n_jobs=-1, random_state=self.random_state, scoring='roc_auc')
        clf_lr.fit(self.X_train, self.y_train)
        return clf_lr.best_estimator_

    def tune_gradient_boosting(self, param_grid):
        gb = GradientBoostingClassifier()
        clf_gb = RandomizedSearchCV(gb, param_grid, cv=self.cv, n_iter=10, n_jobs=-1, random_state=self.random_state, scoring='roc_auc')
        clf_gb.fit(self.X_train, self.y_train)
        return clf_gb.best_estimator_

    def tune_support_vector_classifier(self, param_grid):
        svc = SVC(probability=True)
        clf_svc = RandomizedSearchCV(svc, param_grid, cv=self.cv, n_iter=10, n_jobs=-1, random_state=self.random_state, scoring='roc_auc')
        clf_svc.fit(self.X_train, self.y_train)
        return clf_svc.best_estimator_

    def tune_neural_network(self, param_grid):
        # Create a MLPClassifier
        mlp = MLPClassifier()
        # Use GridSearchCV to tune the model
        clf_mlp = RandomizedSearchCV(mlp, param_grid, cv=self.cv, n_iter=10, n_jobs=-1, random_state=self.random_state, scoring='roc_auc')
        clf_mlp.fit(self.X_train, self.y_train)
        return clf_mlp.best_estimator_


