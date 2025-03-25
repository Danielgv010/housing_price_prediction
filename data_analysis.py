#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from datetime import datetime
from imblearn.over_sampling import SMOTE


# In[223]:


pd.set_option('display.max_rows', None)


# In[224]:


def generate_chunk_html(df_chunk):
    html = df_chunk.to_html(classes='table table-striped', index=False)
    return html

def display_scrollable_table(df, chunk_size=100):
    html_output = "<div style='overflow: auto; max-height: 400px; width:100%;'>"
    for i in range(0, len(df), chunk_size):
        df_chunk = df[i:i + chunk_size] 
        html_output += generate_chunk_html(df_chunk)

    html_output += "</div>"
    display(HTML(html_output))


# In[225]:


# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("data/raw/train.csv")


# In[226]:


display_scrollable_table(df)


# In[227]:


df.isnull().sum().sort_values(ascending=False)


# In[228]:


df.nunique().sort_values(ascending=False)


# In[229]:


for column in df:
    print(f"Unique values for column '{column}':")
    print(df[column].unique())
    print("-" * 30)


# In[230]:


for column in df.isnull().sum().sort_values(ascending=False).index:
    print(f"Unique values for column '{column}':")
    print(df[column].unique())
    print("-" * 30)


# In[231]:


for col in ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']:
    df[col] = df[col].fillna('No' + col[:-2])  #Dynamic "NoPool", "NoMisc" etc

# Masonry Veneer
df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = np.where(df['MasVnrType'] == 'None', 0,
                            df['MasVnrArea'].fillna(
                                df['MasVnrArea'].median()))  # Impute median if not None


# Garage-Related Columns
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    df[col] = df[col].fillna('NoGarage')
for col in ['GarageYrBlt', 'GarageCars', 'GarageArea']:
    df[col] = df[col].fillna(0)


# Basement-Related Columns
for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2']:
    df[col] = df[col].fillna('NoBasement')
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            'BsmtFullBath', 'BsmtHalfBath']:
    df[col] = df[col].fillna(0)


# LotFrontage (Grouped Median Imputation)
df['LotFrontage'] = df['LotFrontage'].fillna(
    df.groupby('Neighborhood')['LotFrontage'].transform('median'))

# If still missing LotFrontage after grouping (rare neighborhoods)
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())


# Electrical (Mode Imputation)
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# Drop the ID column since it's not needed anymore
df = df.drop('Id', axis=1)


# In[232]:


#Verify no missing values left
df.isnull().sum().sort_values(ascending=False).head(10)


# In[233]:


# Histograms of all numeric columns
for column in ['SalePrice']:
    df[column].hist(bins=100)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column}")
    plt.show()


# In[234]:


# Boxplot of price and area
for column in ['SalePrice']:
    df.boxplot(column)
    plt.show()


# In[ ]:


def replace_outliers_iqr(df, column, iqr_factor=1.5):

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR

    # Create a copy to avoid modifying the original DataFrame directly
    df_copy = df.copy()

    # Replace outliers with the bounds
    df_copy[column] = np.where(df_copy[column] < lower_bound, lower_bound, df_copy[column])
    df_copy[column] = np.where(df_copy[column] > upper_bound, upper_bound, df_copy[column])

    return df_copy


#Apply IQR outlier removal to ALL numerical columns *before* scaling and splitting.
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in ['SalePrice']:
    df = replace_outliers_iqr(df, col) # replace outliers with IQR bounds


# In[236]:


# Boxplot of price and area
for column in ['SalePrice']:
    df.boxplot(column)
    plt.show()


# In[237]:


# Heatmap of correlation matrix
correlation_matrix = df.select_dtypes(include=['int64', 'float64']).corr()

plt.figure(figsize=(30, 20))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Correlation Heatmap')
plt.show()


# In[238]:


def encode_and_one_hot(df, columns_to_encode):
    df = df.copy()
    for col in columns_to_encode:
        df[col] = df[col].astype("category").cat.codes

    df = pd.get_dummies(df, columns=columns_to_encode, dummy_na=False)

    return df


# In[239]:


# List of columns to encode
columns_for_encoding = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType',
                           'FireplaceQu', 'GarageQual', 'GarageFinish', 'GarageType',
                           'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtCond',
                           'BsmtQual', 'BsmtFinType1', 'Electrical', 'Condition2',
                           'BldgType', 'Neighborhood', 'LandSlope', 'LotConfig',
                           'Condition1', 'LandContour', 'LotShape', 'Street', 'MSZoning',
                           'Utilities', 'HouseStyle', 'Foundation', 'ExterQual',
                           'ExterCond', 'Heating', 'KitchenQual', 'Functional',
                           'PavedDrive', 'SaleType', 'SaleCondition', 'Exterior1st',
                           'Exterior2nd', 'RoofStyle', 'RoofMatl', 'CentralAir', 'HeatingQC']

# Numerical columns to be treated as categorical (added to encoding list)
columns_for_encoding.extend(['MSSubClass', 'OverallCond', 'OverallQual', 'MoSold',
                             'YrSold', 'KitchenAbvGr', 'FullBath', 'HalfBath',
                             'BsmtFullBath', 'BsmtHalfBath', 'BedroomAbvGr',
                             'Fireplaces', 'GarageCars', 'TotRmsAbvGrd', 'PoolArea'])


# In[240]:


df = encode_and_one_hot(df, columns_for_encoding)


# In[241]:


#fill NA with median
numerical_cols_with_na = [col for col in df.columns if df[col].isnull().any() and df[col].dtype in ['int32', 'float32', 'int64', 'float64']]
for col in numerical_cols_with_na:
    df[col] = df[col].fillna(df[col].median())


# In[242]:


df.to_csv('data/processed/train.csv', index=False)

