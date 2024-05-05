# ML1-50startupprofitprediction-

import pandas as pd
startups = pd.read_csv("/content/drive/MyDrive/50_Startups.csv")
startups.head()

#Take a quick look at The data

startups.head()

startups.info()

startups.isnull().mean() * 100

startups.describe()

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import plotly.express as px

px.scatter(startups, x="R&D Spend", y="Marketing Spend", color="Profit",  title="Startups Profit", width=800, height=600)

startups.hist(bins=50, figsize=(20,15));

#Create a Test Set

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(startups, test_size=0.2, random_state=42)

train_set.shape, test_set.shape

train_set.hist(bins=50,figsize=(20,15));

train_set.Profit.describe(),test_set.Profit.describe()

px.scatter(train_set, x="R&D Spend", y="Marketing Spend", color="Profit",  title="Startups Profit", width=800, height=600)

#EDA

copy=startups.drop("State", axis='columns')

copy.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1])

copy.corr()

copy.corr()["Profit"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix

attributes = ["R&D Spend", "Marketing Spend", "Profit"]
scatter_matrix(train_set[attributes], figsize=(12, 8));

sns.pairplot(train_set);

px.scatter(train_set, x="R&D Spend" , y="Profit"  ,title="Startups Profit" , width=800 , height=600)

#Feature Engineering

# Totalspending per profit
copy["Total_spending_per_profit"] = (copy["R&D Spend"]+train_set["Marketing Spend"])/copy["Profit"]
[114]
0s
copy.corr()["Profit"].sort_values(ascending=False)
Profit                       1.000000
R&D Spend                    0.972900
Marketing Spend              0.747766
Administration               0.200717
Total_spending_per_profit    0.189661
Name: Profit, dtype: float64
R&D Spend is more informative than Total_spending_per_profit or Marketing Spend.

Marketing Spend is more informative Total_spending_per_profit

Total_spending_per_profit is less informative than Administration.

Prepare the Data for Machine Learning Algorithms

# Split the data into features and labels

train_features = train_set.drop("Profit", axis=1)
train_labels = train_set["Profit"].copy()

#Handling Text and Categorical Attributes

startups_cat = train_features[["State"]]
startups_cat.head(10)

startups_cat["State"].value_counts()

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
startups_cat_1hot = cat_encoder.fit_transform(startups_cat)
startups_cat_1hot # returns a sparse matrix

# Convert sparse matrix to a dense matrix
startups_cat_1hot.toarray()

# Make 'sparse=False' to get a dense matrix
cat_encoder = OneHotEncoder(sparse=False)
startups_cat_1hot = cat_encoder.fit_transform(startups_cat)
startups_cat_1hot

cat_encoder.categories_

#Feature Scaling

# Remove the text attribute because median can only be calculated on numerical attributes
startups_num = train_features.drop("State", axis=1)

startups_num.describe().loc[["min", "max"]]

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
startups_num_scaled = scaler.fit_transform(startups_num)
startups_num_scaled

startups_num_scaled.min(axis=0), startups_num_scaled.max(axis=0)

# MinMaxScaler with custom range
custom_scaler = MinMaxScaler(feature_range=(-1, 1))
startups_num_custom_scaled = custom_scaler.fit_transform(startups_num)
startups_num_custom_scaled

startups_num_custom_scaled.min(axis=0), startups_num_custom_scaled.max(axis=0)

# StandardScaler
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
startups_num_std_scaled = std_scaler.fit_transform(startups_num)
startups_num_std_scaled

startups_num_std_scaled.mean(axis=0), startups_num_std_scaled.std(axis=0)

#Transformation Pipelines

# num_pipeline

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([   ('imputer', SimpleImputer(strategy="median")),
                            ('std_scaler', StandardScaler()),
                        ])
startups_num_tr = num_pipeline.fit_transform(startups_num)

# using make_pipeline
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(   SimpleImputer(strategy="median"),
                                StandardScaler()
                            )

startups_num_tr = num_pipeline.fit_transform(startups_num)

# Combine the numerical and categorical pipelines

from sklearn.compose import ColumnTransformer
num_attribs = list(startups_num)
cat_attribs = ["State"]

full_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attribs),
                                    ("cat", OneHotEncoder(sparse= False, drop = 'first'), cat_attribs),
                                  ])

# Using make_column_transformer
from sklearn.compose import make_column_transformer
full_pipeline = make_column_transformer(    (num_pipeline, num_attribs),
                                            (OneHotEncoder(sparse= False), cat_attribs)
                                       )

# Transform the training data
train_features_prepared = full_pipeline.fit_transform(train_features)

train_features_prepared.shape

full_pipeline = make_column_transformer(    (num_pipeline, num_attribs),
                                            (OneHotEncoder(sparse= False, drop='first'), cat_attribs)
                                       )

# Transform the training data
train_features_prepared = full_pipeline.fit_transform(train_features)
train_features_prepared.shape

# Final Pipeline

num_pipeline = Pipeline([   ('imputer', SimpleImputer(strategy="median")),
                            ('std_scaler', StandardScaler()),
                        ])

num_attribs = list(startups_num)
cat_attribs = ["State"]

full_pipeline = ColumnTransformer([ ("num", num_pipeline, num_attribs),
                                    ("cat", OneHotEncoder(sparse= False, drop = 'first'), cat_attribs),
                                  ])

train_features_prepared = full_pipeline.fit_transform(train_features)

# access one hot encoder categories
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_encoder.categories_

train_features_prepared

Researchanddevelopment_Spending = train_features_prepared[:,5]
Researchanddevelopment_Spending

Profit= train_labels.values
Profit

# draw a scatter plot with regression line
sns.regplot(x=Researchanddevelopment_Spending, y=Profit, scatter_kws={"color": "blue"}, line_kws={"color": "red"});

# Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(train_features_prepared, train_labels)

# Predictions
lin_reg_predictions = lin_reg.predict(train_features_prepared)
Decision Tree

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
tree.fit(train_features_prepared, train_labels)

# Predictions
tree_predictions = tree.predict(train_features_prepared)

# Training Accuracy
from sklearn.metrics import r2_score
print("Linear Regression Accuracy: ", r2_score(train_labels, lin_reg_predictions))
print("Decision Tree Accuracy: ", r2_score(train_labels, tree_predictions))

# Cross Validation for Linear Regression

from sklearn.model_selection import cross_val_score
lin_reg_scores = cross_val_score(lin_reg, train_features_prepared, train_labels, scoring="r2", cv=10)

print("Linear Regression Accuracy: ", lin_reg_scores)
print("Linear Regression Accuracy: ", round(lin_reg_scores.mean(),2))
print("Linear Regression Standard Deviation: ", round(lin_reg_scores.std(),2))

# Cross Validation for Decision Tree
from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree, train_features_prepared, train_labels, scoring="r2", cv=10)

print("Scores:", tree_scores)
print("Mean:", round(tree_scores.mean(),2))
print("Standard deviation:", round(tree_scores.std(),2))

#Fine-Tune Your Model

# Find the best model using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [2, 3] , 'max_features': [2, 4,6,7]}
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='r2')
grid_search.fit(train_features_prepared, train_labels)
grid_search.best_params_

# Find the best model using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [2, 3, 4, 5] , 'max_features': [2, 4, 6, 8]}
grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_features_prepared, train_labels)
grid_search.best_params_

# Cross Validation for Decision Tree
tree_best = DecisionTreeRegressor(** grid_search.best_params_)
tree_best.fit(train_features_prepared, train_labels)

# Predictions
tree_bestpredictions = tree_best.predict(train_features_prepared)

# Testing Accuracy
test_features = test_set.drop("Profit", axis=1)
test_features["Total_spending_per_profit"] = (test_set["R&D Spend"]+test_set["Marketing Spend"])/test_set["Profit"]
test_labels = test_set["Profit"].copy()

test_features_prepared = full_pipeline.transform(test_features)

tree_best_predictions_test = tree_best.predict(test_features_prepared)

print("Decision Tree Accuracy on Test Data: ",  r2_score(test_labels, tree_best_predictions_test))

# Testing Accuracy
test_features = test_set.drop("Profit", axis=1)
test_features["Total_spending_per_profit"] = (test_set["R&D Spend"]+test_set["Marketing Spend"])/train_set["Profit"]
test_labels = test_set["Profit"].copy()

test_features_prepared = full_pipeline.transform(test_features)
lin_reg_predictions_test = lin_reg.predict(test_features_prepared)
tree_best_predictions_test = tree_best.predict(test_features_prepared)

print("Decision Tree Accuracy on Test Data: ", r2_score(test_labels, tree_best_predictions_test))

# Linear Regression vs. Decision Tree Training and Testing Accuracy
pd.DataFrame({ "Linear Regression": [ r2_score(test_labels, lin_reg_predictions_test)],
                "Decision Tree": [ r2_score(test_labels, tree_best_predictions_test)]},
                index=["Testing Accuracy"])

# save linear regression model
import joblib
joblib.dump(lin_reg, "lin_reg.pkl")

#Save Your Model

# Load the model
lin_reg = joblib.load("lin_reg.pkl")

# Save Pipeline
import joblib
joblib.dump(full_pipeline, "full_pipeline.pkl")

#Web App
# Use the model to make predictions
lin_reg_predictions_test = lin_reg.predict(test_features_prepared)

# Save Pipeline
import joblib
joblib.dump(full_pipeline, "full_pipeline.pkl")

Web App

%%writefile app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
lin_reg = joblib.load("lin_reg.pkl")

# Load the pipeline
full_pipeline = joblib.load("full_pipeline.pkl")

# Load the data
housing = pd.read_csv("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv")

# Create a title and sub-title
st.title("California Housing Price Prediction App")

st.write("""
This app predicts the **California Housing Price**!
""")

# Take the input from the user
longitude = st.slider('longitude', float(housing['longitude'].min()), float(housing['longitude'].max()))
latitude = st.slider('latitude', float(housing['latitude'].min()), float(housing['latitude'].max()))

housing_median_age = st.slider('housing_median_age', float(housing['housing_median_age'].min()), float(housing['housing_median_age'].max()))
total_rooms = st.slider('total_rooms', float(housing['total_rooms'].min()), float(housing['total_rooms'].max()))
total_bedrooms = st.slider('total_bedrooms', float(housing['total_bedrooms'].min()), float(housing['total_bedrooms'].max()))
population = st.slider('population', float(housing['population'].min()), float(housing['population'].max()))
households = st.slider('households', float(housing['households'].min()), float(housing['households'].max()))
median_income = st.slider('median_income', float(housing['median_income'].min()), float(housing['median_income'].max()))

ocean_proximity = st.selectbox('ocean_proximity', ('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))

# Store a dictionary into a variable
user_data = {'longitude': longitude,

'latitude': latitude,
'housing_median_age': housing_median_age,
'total_rooms': total_rooms,
'total_bedrooms': total_bedrooms,
'population': population,
'households': households,
'median_income': median_income,
'ocean_proximity': ocean_proximity}

# Transform the data into a data frame
features = pd.DataFrame(user_data, index=[0])

# Additional transformations
features['rooms_per_household'] = features['total_rooms']/features['households']
features['bedrooms_per_room'] = features['total_bedrooms']/features['total_rooms']
features['population_per_household'] = features['population']/features['households']

# Pipeline
features_prepared = full_pipeline.transform(features)

# Predict the output
prediction = lin_reg.predict(features_prepared)[0]

# Set a subheader and display the prediction
st.subheader('Prediction')
st.markdown('''# $ {} '''.format(round(prediction), 2))

!streamlit run app.py
