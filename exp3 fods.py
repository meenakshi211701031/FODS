#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_excel('expppp.xlsx')


# In[3]:


df


# In[4]:


df.head(10)


# In[5]:


df.dtypes


# In[19]:


summary=df.describe(include='all')


# In[20]:


summary


# In[21]:


median=df.median(numeric_only=True)


# In[22]:


median


# In[23]:


mode=df.mode(numeric_only=True)


# In[24]:


mode


# In[27]:


missing_value=df.isnull().sum()


# In[28]:


missing_value


# In[29]:


percentage=(missing_value/len(df))*10


# In[30]:


percentage


# In[31]:


# Data Visualization


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt


# In[11]:


plt.figure(figsize=(10,10))


# In[12]:


df=pd.read_excel('expppp.xlsx')


# In[13]:


df


# In[20]:


plt.hist(df['Sale_Price'],bins=30,color='Green',edgecolor='Red')
plt.title("Distribution of sale prices")
plt.xlabel("Sales Price")
plt.ylabel("Frequency")
plt.show()


# In[23]:


import pandas as pd
import seaborn as sns


# In[24]:


df=pd.read_excel('expppp.xlsx')


# In[26]:


sns.boxplot(data=df, y='ntity_In_St')
plt.title('Box Plot of Quantity In Stock')
plt.ylabel('Quantity In Stock')
plt.show()


# In[27]:


Q1 = df['ntity_In_St'].quantile(0.25)
Q3 = df['ntity_In_St'].quantile(0.75)
IQR = Q3 - Q1


# In[28]:


outliers = df[(df['ntity_In_St'] < Q1 - 1.5 * IQR) | (df['ntity_In_St'] > Q3 + 1.5 * IQR)]


# In[30]:


outliers


# In[31]:


plt.scatter(df['rchase_Pri'], df['Sale_Price'], alpha=0.5)
plt.title('Scatter Plot of Purchase Price vs. Sale Price')
plt.xlabel('Purchase Price')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()


# In[32]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap of Correlation Between Numeric Variables')
plt.show()


# In[33]:


correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()


# In[1]:


#Handling Missing Data and Outliers


# In[3]:


import pandas as pd


# In[4]:


df=pd.read_excel('expppp.xlsx')


# In[6]:


df['rchase_Pri'].fillna(df['rchase_Pri'].median(), inplace=True)


# In[7]:


df['Sale_Price'].fillna(df['Sale_Price'].median(), inplace=True)


# In[8]:


print(df[['rchase_Pri', 'Sale_Price']].isnull().sum())


# In[9]:


Q1 = df['Sale_Price'].quantile(0.25)
Q3 = df['Sale_Price'].quantile(0.75)
IQR = Q3 - Q1


# In[10]:


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[11]:


df_outliers_removed = df[(df['Sale_Price'] >= lower_bound) & (df['Sale_Price'] <= upper_bound)]


# In[12]:


print("Summary before removing outliers:")
print(df['Sale_Price'].describe())


# In[13]:


print("\nSummary after removing outliers:")
print(df_outliers_removed['Sale_Price'].describe())


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[15]:


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns[:5]


# In[16]:


sns.pairplot(df[numeric_columns])
plt.show()


# In[17]:


avg_sale_price_by_category = df.groupby('Category')['Sale_Price'].mean().reset_index()


# In[18]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Sale_Price', data=avg_sale_price_by_category)
plt.title('Average Sale Price by Category')
plt.xticks(rotation=45)
plt.show()


# In[20]:


total_quantity_by_supplier = df.groupby('Supplier')['ntity_In_St'].sum().reset_index()


# In[21]:


plt.figure(figsize=(10, 6))
sns.barplot(x='Supplier', y='ntity_In_St', data=total_quantity_by_supplier)
plt.title('Total Quantity In Stock by Supplier')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




