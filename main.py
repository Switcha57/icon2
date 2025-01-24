import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split

from sklearn.preprocessing import LabelEncoder, RobustScaler

from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score

red = pd.read_csv("./original_dataset/Red.csv")
white = pd.read_csv("./original_dataset/White.csv")
sparkling = pd.read_csv("./original_dataset/Sparkling.csv")
rose = pd.read_csv("./original_dataset/Rose.csv")

