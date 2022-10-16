#!/usr/bin/env python
# coding: utf-8

# In[17]:


import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)


# In[18]:


def wrangle(filepath):
    #read csv file into dataframe
    df = pd.read_csv(filepath)
    
    #Subset to properties in '"Madrid"'
    mask_ba = df["province"].str.contains("Madrid")
    
    
    #Subset to properties where '"price_aprox_usd"' < 10000
    mask_price = df["price_usd"] < 10_000
    #Subset
    df = df[mask_ba & mask_price]
    
    #Remove outliers by "Square_meters"
    low, high = df["Square_meters"].quantile([0.1, 0.9])
    mask_area = df["Square_meters"].between(low, high)
    
    df = df[mask_area]


    return df


# In[19]:


df = wrangle("/Users/steve/Desktop/Analytics/project_0/rent_spain_scraping_dataset_main_retransformed.csv")
print("df shape:", df.shape)
df.head()


# In[20]:


#histogram of "Square_meters"
plt.hist(df["Square_meters"])
plt.xlabel("Area [sq meters]")
plt.title("Distribution of Apartment Sizes");


# In[21]:


#Calculate the summary statistics
df.describe()["Square_meters"]


# In[22]:


#Scatter plot that shows price ("price_usd") vs area ("Square_meters") 
plt.scatter(x=df["Square_meters"], y=df["price_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Madrid: Price vs. Area");


# In[23]:


#Model Building
#Create the feature matrix named `X_train`
features = ["Square_meters"]
X_train = df[features]
X_train.shape


# In[24]:


#Create the target vector named y_train
target = "price_usd"
y_train = df[target]
y_train.shape


# In[25]:


#Calculate the mean of the target vector y_train and assign it to the variable y_mean
y_mean = y_train.mean()
y_mean


# In[28]:


#Create a list named y_pred_baseline that contains the value of y_mean repeated so that it's the same length at y
y_pred_baseline = [y_mean] * len(y_train)


# In[29]:


#Calculate the baseline mean absolute error for your predictions in y_pred_baseline as compared to the true targets in y
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))


# In[30]:


#This information tells us that if we always predicted that an apartment price is $1355.55, our predictions would be off by an average of $453.83. It also tells us that our model needs to have mean absolute error below $453.83 in order to be useful.


# In[31]:


model = LinearRegression()


# In[32]:


model.fit(X_train, y_train)


# In[33]:


#create a list of predictions for the observations in your feature matrix X_train
y_pred_training = model.predict(X_train)
y_pred_training[:5]


# In[34]:


#Calculate training mean absolute error for your predictions
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))


# In[35]:


#Extract the intercept from the model
intercept = round(model.intercept_, 2)
print("Model Intercept:", intercept)
assert any([isinstance(intercept, int), isinstance(intercept, float)])


# In[36]:


#Extract the coefficient associated "surface_covered_in_m2"
coefficient = round(model.coef_[0], 2)
print('Model coefficient for "Square_meters":', coefficient)
assert any([isinstance(coefficient, int), isinstance(coefficient, float)])


# In[37]:


#print the equation of the model
print(f"apt_price = {coefficient} * surface_covered")


# In[ ]:




