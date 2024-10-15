#!/usr/bin/env python
# coding: utf-8

# #### Import Libraries

# In[2]:


#import bamboolib as bam 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import statsmodels.api as sm  
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import math
from random import uniform
from scipy.stats import  randint as sp_randint
import urllib.request
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')


# ## Selecting the directory

# In[3]:


pwd


# ## Importing the dataset

# In[4]:


dataframe=pd.read_csv("Housing_data.csv")
dataframe.head()


# In[5]:


dataframe.shape


# #### Notes: The Seattle King County dataset has 21613 rows and 21 columns

# In[6]:


dataframe.info() 


# In[7]:


dataframe.columns


# In[8]:


# Checking the Data types
dataframe.dtypes


# In[9]:


# Summary 
dataframe.describe()


# #### Note: It is more common for houses built in 2015 or on an average to have two bathrooms. Although there are some variables to worry about (the maximum number of bedrooms in a house is 33 and the maximum number of bathrooms is 8).

# ## Checking for missing values

# In[10]:


# Percentage of missing values per column
(dataframe.isna().sum() / len(dataframe)) * 100


# ## Checking for duplicates

# In[11]:


# Check to see if 'id' is unique identifier for each sample
print('Sum of duplicate values:{}\n'.format(dataframe.id.duplicated().sum()))


# In[12]:


dataframe.loc[dataframe.id.duplicated(), :]


# In[13]:


# Drop duplicates 
dataframe = dataframe.sort_values('id', ascending = False).drop_duplicates(subset = 'id', keep = 'first')


# In[14]:


# Checking again for duplicates
dataframe['id'].duplicated().sum()


# In[15]:


dataframe.shape


# In[16]:


duplicate = dataframe[dataframe.duplicated()]  
print("Duplicate Rows :",duplicate)


# ## Data Cleaning

# #### There is some redundant data in the Date column. There is no use for the H:M:S part of the data since it is always 'T000000'. Thus, the H:M:S will be cleaned up. Transforming date columns into years, months, and days

# ### a) Column 'date'

# In[17]:


dataframe['date'] = pd.to_datetime(dataframe['date'])
dataframe['year'] = dataframe['date'].dt.year
dataframe['month'] = dataframe['date'].dt.month
dataframe['day'] = dataframe['date'].dt.day
dataframe.head()


# ### b) Column 'bedrooms'

# In[18]:


# Checking Bedrooms column
dataframe['bedrooms'].value_counts()


# In[19]:


dataframe.loc[dataframe['bedrooms'] >= 10]


# In[20]:


# drop row with 33 bedrooms
dataframe.drop(index=15870,inplace = True)


# ## Feature Engineering

# In[21]:


# Drop date and id column as we converted date to year,month,day
dataframe = dataframe.drop("id",axis=1)
dataframe = dataframe.drop("date",axis=1)


# In[22]:


# Calculating the Age of the House
from datetime import datetime, date
dataframe['age']= 2015- dataframe['yr_built']
dataframe.head()


# In[23]:


bin_age = [-2,0,5,10,25,50,75,100,100000]
labels = ['<1','1-5','6-10','11-25','26-50','51-75','76-100','>100']

dataframe['age_group'] = pd.cut(dataframe['age'], bins = bin_age, labels = labels)
dataframe.head()


# In[24]:


# Price per sqft
dataframe['price_per_sqft'] = dataframe['price'] / dataframe['sqft_lot']
dataframe.head()


# In[25]:


bin_price =[0,250000,500000,750000, 1000000, 2000000, dataframe['price'].max()]
label_price = ['upto 250k', 'upto 500k', 'upto 750k','upto 1mil','upto 2mil', 'more than 2 mil']

dataframe['price_group'] = pd.cut(dataframe['price'], bins = bin_price, labels = label_price)
dataframe.head()


# In[26]:


dataframe.info() 


# ## BoxPlot: Checking and Handling Outliers

# In[27]:


# Boxplot to check distribution in each column

dataframe_describe = dataframe.drop(columns = ['lat','long', 'zipcode','price_per_sqft','price_group','age_group'])

plt.figure(figsize =(20, 20))
x = 1 

for column in dataframe_describe.columns:
    plt.subplot(4,5,x)
    sns.boxplot(dataframe_describe[column])
    x+=1
plt.tight_layout
plt.show()


# In[28]:


### Checking the value that's unsual bedroom and bathroom that's 0 

dataframe[dataframe['bedrooms'] < 1]

# There are rows that does not have bedrooms, therefore I will treat this like a null value and fill it with median numbers of bedroom in a house 
# same goes to the bathrooms with 0 values as well 

# Since House with no bathroom / bedrooms is kind of unsual I will replace the 0 value in both of the columns with median
# I used median instead of mean because the data distribution is skewed

dataframe['bedrooms'] = dataframe['bedrooms'].replace(0, dataframe['bedrooms'].median())
dataframe['bathrooms'] = dataframe['bathrooms'].replace(0, dataframe['bathrooms'].median())



# In[29]:


# Rechecking bedrooms
dataframe[dataframe['bedrooms'] < 1]


# ## Exploratory Data Analysis 

# In[30]:


# Visualizing the distribution 
dataframe.hist(figsize=(20,18),bins=10, grid=False)
plt.show()


# In[31]:


plt.figure(figsize = (10, 7))
sns.distplot(dataframe['price'])

# Majority of the prices are  less than 1000000 $ (right skewed)
# this distribution plot does not tell much about anything


# In[32]:


plt.figure(figsize = (15, 8))

sns.countplot(dataframe['age_group'],palette = "afmhot")
plt.title('Count of Houses per Age Group')
plt.show()


# In[33]:


plt.figure(figsize = (15, 8))

sns.countplot(dataframe['price_group'], palette = "Dark2_r")
plt.xticks(rotation = 45)
plt.title('Count of Houses per Price Group')
plt.show()


# In[34]:


plt.figure(figsize = (18, 8))
plt.subplot(1,2,2)
sns.countplot(dataframe['bathrooms'], palette = "YlOrRd_r")
plt.xticks(rotation = 90)
plt.title('Number of Bathrooms')
plt.subplot(1,2,1)
sns.countplot(dataframe['bedrooms'], palette = "YlOrRd_r")
plt.title('Number of Bedrooms')



# Buyer in Seattle preffered a house that has 3 to 4 bedrooms and 1 to 2.5 bathrooms


# In[35]:


plt.figure(figsize = (10, 7))
sns.countplot(dataframe['view'], palette = 'Blues_r')
plt.title('Count of house based on view rating')

# 90 % of the house sold has 0 view 
# this shows that buyer in Seattle doesn't really care about view rating of a house 


# ## HeatMap

# In[36]:


plt.figure(figsize=(15,8))
sns.heatmap(dataframe.corr(), annot=True, cmap='Oranges')


# In[37]:


correlation = dataframe.corr()
correlation['price'].sort_values(ascending = False)[1:]


# In[38]:


sns.countplot(dataframe['waterfront'], palette = 'BuGn_r')

# Almost all of the house in this dataset doesn't have a front facing waterfront
# Waterfront is not a feature that many buyer considered before buying a home in Seattle


# In[39]:


plt.figure(figsize = (10,7))
sns.barplot(x = 'waterfront', y='price_group', data = dataframe, ci = False, palette= 'inferno')

### The more expensive the houses the more likely the house has a waterfront


# In[40]:


color_scale = [(0, 'orange'), (1,'red')]

fig = px.scatter_mapbox(dataframe, 
                        lat="lat", 
                        lon="long", 
                        hover_name="price", 
                        hover_data=dataframe.columns, 
                        color="price",
                        color_continuous_scale=color_scale,
                        size="sqft_living",
                        zoom=8, 
                        height=800,
                        width=800)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[41]:


plt.figure(figsize = (10 , 7))
sns.barplot(x = dataframe['age'], y= dataframe['price_group'], ci = False, palette = 'icefire_r')
plt.title('Average Age of the house per Price Group')
plt.xticks(rotation = 0)
plt.show()
# the age group of the house doesn't really have a positve or negative correlation with the price
# house that's on the lower price tend to be older in age 


# In[42]:


print("Grade counting description")
print(dataframe['grade'].value_counts())


# In[43]:


plt.figure(figsize = (10, 7))
sns.countplot(dataframe['grade'], palette = 'flare')
plt.title('Count based on House Grade')


# In[93]:


plt.figure(figsize = (10 , 7))
sns.barplot(y = dataframe['sqft_basement'], x= dataframe['price_group'], ci = False, palette = 'plasma')
plt.title('Sqft Basement of the house per Price Group')
plt.xticks(rotation = 0)
plt.show()


# In[ ]:


plt.figure(figsize = (10 , 7))
sns.barplot(y = dataframe['grade'], x= dataframe['price_group'], ci = False, palette = 'icefire_r')
plt.title('Grade of the house per Price Group')
plt.xticks(rotation = 0)
plt.show()


# In[45]:


# Average house prices by zipcode

# Function to output average price for a given zipcode
def zip_avg(zipcode):
    zip_avg = []
    for _zip in zipcode:
        _zip = dataframe[(dataframe['zipcode'] == _zip)]
        zip_avg.append(_zip['price'].mean())
    return (zip_avg)


# In[46]:


# Extracting unique zip codes from the dataset
all_zipcodes = dataframe.zipcode.unique()
# Extracting average price for each zipcode
zipcode_average = list(zip_avg(all_zipcodes))

# Creating the dataframe of average price and their corrsponding zipcode
price_by_zipcode = pd.DataFrame([])
price_by_zipcode['Zipcodes'] = all_zipcodes
price_by_zipcode["Average_house_price"] = zipcode_average

#sorting by price
price_by_zipcode = price_by_zipcode.sort_values(by=['Average_house_price']).reset_index(drop=True)
price_by_zipcode


# In[47]:


plt.figure(figsize=(22,10))
plt.xticks(rotation=70, fontsize=13)
plt.yticks(fontsize=15)
plt.title('Average house price by Zipcode', fontsize=20)
plt.xlabel('Zipcodes', fontsize=18)
plt.ylabel('Avg',fontsize=18)
sns.barplot(y=price_by_zipcode['Average_house_price'], x=price_by_zipcode['Zipcodes'], order=price_by_zipcode['Zipcodes'], palette = "RdYlGn_r")
plt.show()


# House in zipcode 98039,98004, 98040 are the top 3 zipcode with the highest average price 
# House in zipcode 98002,98168, 98032 are the top 3 zipcode with the lowest average price


# In[48]:


pd.crosstab(index = dataframe['zipcode'], columns = 'Average_Price', values = dataframe['price'], aggfunc = 'mean').sort_values('Average_Price',ascending = False).head(3)


# In[49]:


pd.crosstab(index = dataframe['zipcode'], columns = 'Average_Price', values = dataframe['price'], aggfunc = 'mean').sort_values('Average_Price',ascending = True).head(3)


# In[50]:


plt.figure(figsize = (20, 8))
plt.subplot(1,2,1)
sns.barplot(data = dataframe, y='price', x = 'bedrooms', palette = 'autumn_r', ci = False)
plt.title('Average Prices Per Number of Bedrooms')
plt.subplot(1,2,2)
sns.barplot(data = dataframe, y='price', x = 'bathrooms', palette = 'autumn_r', ci = False)
plt.xticks(rotation = 90)
plt.title('Average Prices Per Number of Bathrooms')


# The more the bedrooms doesn't mean the more expensive the houses
# House with 8 bedrooms shows to be the most expensive house in average 
# Number of bathrooms somewhat have a positive correlation with the average price of the houses price 


# In[51]:


plt.figure(figsize = (20, 8))
sns.barplot(x = 'grade', y='price', data = dataframe, ci = False, palette = 'Accent_r')
plt.title('Average Prices Per Quality')


# In[52]:


dataframe.info() 


# In[53]:


# Monthly sales
# Function to output average price for a given month
def month_avg(month):
    month_avg = []
    for _m in month:
        _m = dataframe[(dataframe['month'] == _m)]
        month_avg.append(_m['price'].mean())
    return (month_avg)


# In[54]:


# Extracting unique zip codes from the dataset
all_months = dataframe.month.unique()
# Extracting average price for each zipcode
monthsales_average = list(month_avg(all_months))

# Creating the dataframe of average price and their corrsponding zipcode
price_by_month = pd.DataFrame([])

price_by_month['Month'] = all_months
price_by_month["Average_house_price"] = monthsales_average

#sorting by price
price_by_month = price_by_month.sort_values(by=['Average_house_price']).reset_index(drop=True)
price_by_month


# In[1]:


fig = px.line(price_by_month.sort_values(by=['Month'], ascending=[True]), x='Month',line_shape="spline")
fig


# In[56]:


dataframe_new=dataframe


# In[57]:


dataframe_new['price'].where(~(dataframe.price < 750000), other=0, inplace=True)
dataframe_new['price'].where(~(dataframe.price >= 750000), other=1, inplace=True)


# In[58]:


sns.countplot(dataframe_new['price'])


# In[59]:


dataframe_new=dataframe.drop(columns = ['lat','long', 'zipcode','price_per_sqft','price_group','age_group'])
dataframe_new=dataframe_new.drop(columns=['yr_built','yr_renovated','year','month','day'])


# ### Checking for Multi-Collinearity

# In[60]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[61]:


dataframe_new= dataframe_new.drop(['sqft_above'], axis=1)
calc_vif(dataframe_new)


# In[62]:


dataframe_new=dataframe_new.drop(columns = ['sqft_living'])
calc_vif(dataframe_new)


# In[63]:


dataframe_new=dataframe_new.drop(columns = ['grade'])
calc_vif(dataframe_new)


# In[64]:


dataframe_new=dataframe_new.drop(columns = ['bathrooms'])
calc_vif(dataframe_new)


# In[65]:


dataframe_new=dataframe_new.drop(columns = ['condition'])
calc_vif(dataframe_new)


# In[66]:


dataframe_new=dataframe_new.drop(columns = ['bedrooms'])
calc_vif(dataframe_new)


# In[ ]:




