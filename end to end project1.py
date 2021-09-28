#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# In[2]:


train_data=pd.read_excel('D:\PROJET1\Data_Train.xlsx')

train_data.head()
# In[3]:


train_data.shape


# In[4]:


train_data.isna().sum()


# In[5]:


#dealingwithmissingvalues
train_data.dropna(inplace=True)


# In[6]:


#after
train_data.isna().sum()


# In[7]:


train_data.dtypes


# In[8]:


train_data.head()


# In[9]:


train_data.info()


# In[10]:


def change_into_datetime(col):
    train_data[col]=pd.to_datetime(train_data[col])


# In[11]:


train_data.columns


# In[12]:


for i in ['Date_of_Journey', 'Dep_Time', 'Arrival_Time']:
    change_into_datetime(i)


# In[13]:


train_data.dtypes


# In[ ]:





# In[14]:


train_data['journey.day']=train_data['Date_of_Journey'].dt.day
train_data['journey.month']=train_data['Date_of_Journey'].dt.month


# In[15]:


train_data.head()


# In[ ]:





# In[16]:


train_data.drop('Date_of_Journey', axis=1, inplace=True)


# In[17]:


def extract_hour(df,col):
    df[col+'_hour']=df[col].dt.hour
    
def extract_min(df,col):
    df[col+'_minute']=df[col].dt.minute
    
def drop_column(df, col):
    df.drop(col,axis=1,inplace=True)


# In[18]:


extract_hour(train_data,'Dep_Time')
extract_min(train_data,'Dep_Time')
drop_column(train_data,'Dep_Time')


# In[19]:


extract_hour(train_data,'Arrival_Time')
extract_min(train_data,'Arrival_Time')
drop_column(train_data,'Arrival_Time')


# In[20]:


train_data.head()

duration=list(train_data['Duration'])
# In[21]:


duration=list(train_data['Duration'])


# In[22]:


x='19h0mn'
x.split('h')[0]


# In[23]:


for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]:
            duration[i]=duration[i]
        else:
            duration[i]='0h '+duration[i]
            


# In[24]:


train_data['Duration']=duration


# In[25]:


train_data.head()


# In[26]:


def hour(x):
    return x.split('h')[0]


# In[27]:


def minute(x):
    return x.split(' ')[-1][0:-1]


# In[28]:


train_data['Duration_hours']=train_data['Duration'].apply(hour)
train_data['Duration_mins']=train_data['Duration'].apply(minute)


# In[29]:


train_data.head()


# In[30]:


drop_column(train_data,'Duration')


# In[31]:


train_data.dtypes


# In[32]:


train_data['Duration_hours']=train_data['Duration_hours'].astype('int')
train_data['Duration_mins']=train_data['Duration_mins'].astype('int')


# In[33]:


train_data.dtypes


# In[ ]:





# In[34]:


cat_col=[col for col in train_data.columns if train_data[col].dtype=='object']
cat_col


# In[35]:


cont_col=[col for col in train_data.columns if train_data[col].dtype!='object']
cont_col


# In[ ]:





# In[36]:


categorical=train_data[cat_col]


# In[37]:


categorical.head()


# In[38]:


categorical['Airline'].value_counts()


# In[ ]:





# In[39]:


plt.figure(figsize=(15,5))
sns.boxplot(x='Airline',y='Price',data=train_data.sort_values('Price',ascending=False))


# In[ ]:





# In[40]:


train_data.head()


# In[ ]:





# In[41]:


plt.figure(figsize=(15,5))
sns.boxplot(x='Total_Stops',y='Price',data=train_data.sort_values('Price',ascending=False))


# In[ ]:





# In[42]:


Airline=pd.get_dummies(categorical['Airline'],drop_first=True)


# In[43]:


Airline.head()


# In[ ]:





# In[44]:


categorical['Source'].value_counts()


# In[ ]:





# In[45]:


plt.figure(figsize=(15,5))
sns.boxplot(x='Source',y='Price',data=train_data.sort_values('Price',ascending=False))


# In[ ]:





# In[46]:


Source=pd.get_dummies(categorical['Source'],drop_first=True)
Source.head()


# In[ ]:





# In[47]:


categorical['Destination'].value_counts()


# In[ ]:





# In[48]:


plt.figure(figsize=(15,5))
sns.boxplot(x='Destination',y='Price',data=train_data.sort_values('Price',ascending=False))


# In[ ]:





# In[49]:


Destination=pd.get_dummies(categorical['Destination'],drop_first=True)
Destination.head()


# In[50]:


categorical.head()


# In[64]:


ROUTE_1=categorical['Route'].str.split('→').str[0]
ROUTE_2=categorical['Route'].str.split('→').str[1]
ROUTE_3=categorical['Route'].str.split('→').str[2]
ROUTE_4=categorical['Route'].str.split('→').str[3]
ROUTE_5=categorical['Route'].str.split('→').str[4]


# In[65]:


categorical.head()


# In[66]:


categorical.columns


# In[67]:


categorical.drop(Route,axis=1,inplace=True)


# In[68]:


categorical.head()


# In[70]:


categorical.isnull().sum()


# In[81]:


for i in ['Route_3', 'Route_4', 'Route_5', 'Total_Stops']:
    categorical[i].fillna('None',inplace=True)


# In[ ]:





# In[82]:


categorical.isnull().sum()


# In[83]:


for i in categorical.columns:
    print('{} has total {} categories'.format(i,len(categorical[i].value_counts())))


# In[112]:


dict = {}
dict["4 stops"] = 4
dict["2 stops"] = 2
dict["1 stop"] = 1
dict["3 stops"] = 3
dict["non-stop"] = 0


# In[86]:


from sklearn.preprocessing import LabelEncoder


# In[87]:


encoder=LabelEncoder()


# In[88]:


categorical.columns


# In[95]:


for i in [ 'Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5']:
    i=encoder.fit_transform(categorical[i])


# In[96]:


categorical.head()


# In[379]:





# In[115]:


categorical['Total_Stops'].unique()


# In[117]:


dict = {}
dict["4 stops"] = 4
dict["2 stops"] = 2
dict["1 stop"] = 1
dict["3 stops"] = 3
dict["non-stop"] = 0


# In[118]:


Total_Stops=categorical['Total_Stops'].map(dict)


# In[119]:


categorical.head()


# In[ ]:





# In[120]:


data_train=pd.concat([categorical.Airline,Source,Destination,train_data[cont_col]],axis=1)


# In[121]:


data_train.head()


# In[137]:


drop_column(data_train, 'Airline')
drop_column(data_train, 'Destination')        


# In[ ]:





# In[138]:


pd.set_option('display.max_columns',35)
data_train.head()


# In[ ]:





# In[139]:


def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)


# In[140]:


plot(data_train,'Price')


# In[ ]:





# In[141]:


data_train['Price']=np.where(data_train['Price']>=40000,data_train['Price'].median(),data_train['Price'])


# In[ ]:





# In[142]:


x=data_train.drop('Price',axis=1)
x.head()


# In[ ]:





# In[143]:


x.shape


# In[144]:


y=data_train['Price']
y


# In[145]:


from sklearn.feature_selection import mutual_info_classif


# In[146]:


mutual_info_classif(x,y)


# In[147]:


imp=pd.DataFrame(mutual_info_classif(x,y),index=x.columns)
imp


# In[ ]:





# In[148]:


imp.columns=['Importance']
imp.sort_values(by='Importance',ascending=False)


# In[149]:


from sklearn.model_selection import train_test_split


# In[150]:


x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2)


# In[152]:


from sklearn import metrics


# In[163]:


import pickle
def predict(ml_model,dump):
    model=ml_model.fit(x_train,y_train)
    print('Training score : {}'.format(model.score(x_train,y_train)))
    y_prediction=model.predict(x_test)
    print('predictions are :\n {}'.format(y_prediction))
    print('\n')
    
    r2_score=metrics.r2_score(y_test,y_prediction)
    print('r2 score is : {}'.format(r2_score))
    
    print('MAE :' , metrics.mean_absolute_error(y_test,y_prediction))
    print('MSE :' , metrics.mean_squared_error(y_test,y_prediction))
    print('RMSE :' , np.sqrt(metrics.mean_squared_error(y_test,y_prediction)))
    
    sns.displot(y_test-y_prediction)
    if dump==1:
        file=open('D:\PROJET1\model.pkl')
        pickle.dump(model.file)


# In[164]:


from sklearn.ensemble import RandomForestRegressor


# In[165]:


predict(RandomForestRegressor(),1)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


# In[158]:


predict( LinearRegression(),0)


# In[161]:


predict( KNeighborsRegressor(),0)


# In[162]:


predict(DecisionTreeRegressor(),0)


# In[ ]:




