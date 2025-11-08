import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer


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


excluded_col = ["ID", "Name", "Addiction_Level"]

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


# In[81]:


train_dict = df_train[categorical + numerical].to_dict(orient="records")
val_dict = df_val[categorical + numerical].to_dict(orient="records")
test_dict = df_test[categorical + numerical].to_dict(orient="records")

X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)


# # Train a Linear Regression Model

# In[82]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error


# In[83]:


lr = LinearRegression()


# In[84]:


lr.fit(X_train, y_train)


# In[85]:


y_pred = lr.predict(X_val)


# In[86]:


'''def root_mean_squared_error(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)'''


# In[87]:


rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE for Linear Regression: {rmse}")


# In[88]:


ridge = Ridge(alpha=1.0, random_state=42)
lasso = Lasso(alpha=1.0, random_state=42)


# In[89]:


ridge.fit(X_train, y_train)


# In[90]:


y_pred = ridge.predict(X_val)

rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE for Ridge: {rmse}")


# # Retrain Ridge with full train dataset

# In[95]:


full_train_dict = df_full_train[categorical + numerical].to_dict(orient="records")

X_full_train = dv.transform(full_train_dict)
y_full_train = df_full_train["Addiction_Level"].values


# In[96]:


ridge.fit(X_full_train, y_full_train)


# In[97]:


y_pred = ridge.predict(X_test)

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for Ridge with full train: {rmse}")


# In[98]:


lasso.fit(X_train, y_train)


# In[99]:


y_pred = lasso.predict(X_val)

rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE for Lasso: {rmse}")


# # Train a Decision Tree Model

# In[100]:


from sklearn.tree import DecisionTreeRegressor


# In[101]:


dt = DecisionTreeRegressor(max_depth=5, min_samples_leaf=50)


# In[102]:


dt.fit(X_train, y_train)


# In[103]:


y_pred = dt.predict(X_val)

rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE for DecisionTreeRegressor: {rmse}")


# # Train a Random Forest Model

# In[104]:


from sklearn.ensemble import RandomForestRegressor


# In[105]:


rf = RandomForestRegressor(
    n_estimators=5,
    #max_depth=3,
    min_samples_leaf=3,
    random_state=1, 
    n_jobs=-1)


# In[106]:


rf.fit(X_train, y_train)


# In[107]:


y_pred = rf.predict(X_val)

rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE for RandomForestRegressor: {rmse}")


# # Gradient boosting and XGBoost

# In[108]:


import xgboost as xgb


# In[109]:


features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


# In[110]:


watchlist = [(dtrain, 'train'), (dval, 'val')]


# In[111]:


get_ipython().run_cell_magic('capture', 'output', "xgb_params = {\n    'eta': 0.3, \n    'max_depth': 6,\n    'min_child_weight': 1,\n\n    'objective': 'reg:squarederror',\n    'nthread': 8,\n\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, \n                  dtrain, \n                  num_boost_round=10, \n                  verbose_eval=5,\n                  evals=watchlist)\n")


# In[112]:


y_pred = model.predict(dval)

rmse = root_mean_squared_error(y_val, y_pred)
print(f"RMSE for XGBoost: {rmse}")


# # Retrain XGBoost Model with Full Train Dataset

# In[117]:


full_train_dict = df_full_train[categorical + numerical].to_dict(orient="records")

X_full_train = dv.transform(full_train_dict)
y_full_train = df_full_train["Addiction_Level"].values


# In[118]:


features = list(dv.get_feature_names_out())
dfull_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)


# In[120]:


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


# In[121]:


y_pred = model.predict(dtest)

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE for XGBoost with full_train dataset: {rmse}")


# # Save Model Locally

# In[122]:


import pickle
#from sklearn.pipeline import make_pipeline


# In[123]:


model_path = "model.bin"

with open(model_path, "wb") as f_out:
    pickle.dump((dv, model), f_out)


# # Test Model

# In[137]:


teen = df.iloc[10][categorical + numerical]
teen


# In[138]:


y_teen_actual = df.iloc[10].Addiction_Level
y_teen_actual


# In[148]:


teen.to_dict()


# In[149]:


teen_dict = {'Gender': 'Female',
             'Location': 'Cherylburgh',
             'School_Grade': '12th',
             'Phone_Usage_Purpose': 'Other',
             'Addiction_Category': 'High',
             'Age': 18,
             'Daily_Usage_Hours': 4.9,
             'Sleep_Hours': 7.0,
             'Academic_Performance': 74,
             'Social_Interactions': 5,
             'Exercise_Hours': 0.6,
             'Anxiety_Level': 4,
             'Depression_Level': 3,
             'Self_Esteem': 2,
             'Parental_Control': 0,
             'Screen_Time_Before_Bed': 0.6,
             'Phone_Checks_Per_Day': 84,
             'Apps_Used_Daily': 20,
             'Time_on_Social_Media': 3.1,
             'Time_on_Gaming': 0.6,
             'Time_on_Education': 0.8,
             'Family_Communication': 6,
             'Weekend_Usage_Hours': 3.5}


# In[150]:


with open("./model.bin", "rb") as f_out:
    dv, model = pickle.load(f_out)


# In[151]:


X_teen = dv.transform(teen_dict)


# In[153]:


features = list(dv.get_feature_names_out())
dteen = xgb.DMatrix(X_teen, feature_names=features)


# In[155]:


y_teen_pred = model.predict(dteen)
print(y_teen_pred)


# In[ ]:




