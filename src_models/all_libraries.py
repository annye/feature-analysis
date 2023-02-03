# All imports libraries

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# import pandas_profiling
# from pandas_profiling import ProfileReport
from pdb import set_trace
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate 

# Shapiro-Wilk Test
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
from scipy import stats
from scipy.stats import normaltest


import scipy as sp
from scipy.optimize import minimize
from scipy.stats import chi2, pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_is_fitted


from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression

 
from collections import Counter
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdb import set_trace
import pickle
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomTreesEmbedding, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.metrics import plot_confusion_matrix, classification_report, plot_precision_recall_curve, plot_roc_curve, balanced_accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, MaxAbsScaler
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display, Markdown
from sklearn.feature_selection import f_regression, SelectKBest


# plt.rcParams["figure.figsize"] = (18.5, 10.5)
# plt.rcParams.update({'font.size': 22})

# importing date class from datetime module
from datetime import date





