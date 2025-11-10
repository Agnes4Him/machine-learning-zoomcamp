import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb


data_path = "https://raw.githubusercontent.com/Agnes4Him/project-datasets/refs/heads/main/teen_phone_addiction_dataset2.csv"
df = pd.read_csv(data_path)


def categorize_level(x):
    if x < 4:
        return "Low"
    elif 4 < x < 7:
        return "Moderate"
    else:
        return "High"

df["Addiction_Category"] = df["Addiction_Level"].apply(categorize_level)


# # Split Dataset

object_cols = df.select_dtypes(include='object').columns
non_object_cols = df.select_dtypes(exclude='object').columns


excluded_col = ["ID", "Name", "Addiction_Level", "Addiction_Category"]

categorical = [c for c in object_cols if c not in excluded_col]
numerical = [c for c in non_object_cols if c not in excluded_col]

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train["Addiction_Level"].values
y_val = df_val["Addiction_Level"].values
y_test = df_test["Addiction_Level"].values

del df_train["Addiction_Level"]
del df_val["Addiction_Level"]
del df_test["Addiction_Level"]

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient="records")
val_dict = df_val[categorical + numerical].to_dict(orient="records")
test_dict = df_test[categorical + numerical].to_dict(orient="records")

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)


features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

watchlist = [(dtrain, 'train'), (dval, 'val')]

xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'reg:squarederror',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, 
                  dtrain, 
                  num_boost_round=10, 
                  verbose_eval=5,
                  evals=watchlist)

y_pred = model.predict(dval)

rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE for XGBoost: {rmse}")


full_train_dict = df_full_train[categorical + numerical].to_dict(orient="records")

X_full_train = dv.transform(full_train_dict)
y_full_train = df_full_train["Addiction_Level"].values

dfull_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)


xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, 
                  dfull_train, 
                  num_boost_round=10, 
                  verbose_eval=5,
                  evals=watchlist)


y_pred = model.predict(dtest)

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for XGBoost with full_train dataset: {rmse}")

model_path = "model.bin"

with open(model_path, "wb") as f_out:
    pickle.dump((dv, model), f_out)





