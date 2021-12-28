'Import libraries:
'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'Load data and provide top 5 samples:
'
addm = pd.read_csv('ADDM.csv')
df = pd.DataFrame(addm)
df.head(5)

'Provide distinct elements:
'
df.nunique(dropna = True)

'Mean:
'
np.mean(df.male_prev.dropna())
np.mean(df.female_prev.dropna())
np.mean(df.white_prev.dropna())
np.mean(df.black_prev.dropna())
np.mean(df.hisp_prev.dropna())
np.mean(df.api_prev.dropna())

'Median:
'
np.median(df.male_prev.dropna())
np.median(df.female_prev.dropna())
np.median(df.white_prev.dropna())
np.median(df.black_prev.dropna())
np.median(df.hisp_prev.dropna())
np.median(df.api_prev.dropna())

'Standard Deviation:
'
np.std(df.male_prev.dropna())
np.std(df.female_prev.dropna())
np.std(df.white_prev.dropna())
np.std(df.black_prev.dropna())
np.std(df.hisp_prev.dropna())
np.std(df.api_prev.dropna())

'Variance:
'
np.var(df.male_prev.dropna()) 
np.var(df.female_prev.dropna()) 
np.var(df.white_prev.dropna()) 
np.var(df.black_prev.dropna()) 
np.var(df.hisp_prev.dropna()) 
np.var(df.api_prev.dropna()) 

'Minimum:
'
print(min(df.male_prev.dropna())) 
print(min(df.female_prev.dropna())) 
print(min(df.white_prev.dropna())) 
print(min(df.black_prev.dropna())) 
print(min(df.hisp_prev.dropna())) 
print(min(df.api_prev.dropna())) 

'Maximum:'
print(max(df.male_prev.dropna())) 
print(max(df.female_prev.dropna())) 
print(max(df.white_prev.dropna())) 
print(max(df.black_prev.dropna())) 
print(max(df.hisp_prev.dropna())) 
print(max(df.api_prev.dropna()))

'Mode:
'
from scipy import stats as st
st.mode(df.male_prev.dropna())
st.mode(df.female_prev.dropna())
st.mode(df.white_prev.dropna())
st.mode(df.black_prev.dropna())
st.mode(df.hisp_prev.dropna())
st.mode(df.api_prev.dropna())

'Conditional statement (dataframe):
'
df.fillna(0, inplace = True)
df['M_prev>than_avg'] = np.where(df['male_prev']> 19.6, True, False)
df['F_prev>than_avg'] = np.where(df['female_prev']> 4.6, True, False)
df['W_prev>than_avg'] = np.where(df['white_prev']> 13.2, True, False)
df['B_prev>than_avg'] = np.where(df['black_prev']>11.1, True, False)
df['H_prev>than_avg'] = np.where(df['hisp_prev']>8.7, True, False)
df['API_prev>than_avg'] = np.where(df['api_prev']>10.2, True, False)
df.head(15)

'Functions:' 
def m_above_avg(x):
    m = x.mean()
    return True if m > x.mean() else False
df.groupby('abbr').male_prev.agg(['count','max','mean','median', m_above_avg])

def w_above_avg(x):
    w = x.mean()
    return True if w > x.mean() else False
df.groupby('abbr').white_prev.agg(['count','max','mean','median', w_above_avg])

def b_above_avg(x):
    b = x.mean()
    return True if b > x.mean() else False
df.groupby('abbr').black_prev.agg(['count','max','mean','median', b_above_avg])

def h_above_avg(x):
    h = x.mean()
    return True if h > x.mean() else False
df.groupby('abbr').hisp_prev.agg(['count','max','mean','median', h_above_avg])

def api_above_avg(x):
    api = x.mean()
    return True if api > x.mean() else False
df.groupby('abbr').api_prev.agg(['count','max','mean','median', api_above_avg])

'Iterative calculation:' 
for i in range(len(df)):
    print(df.iloc[i,0], df.iloc[i,1], df.iloc[i,20])

'Vectors:'
vector1 = [' M ', ' F ', ' W ', 'B ', 'H ', 'API ']
vector2 = [19.63, 4.58, 13.22, 11.12, 8.66, 10.2]
print(vector1)
print(vector2)

'Matrices
:'
vector2 = [19.63, 4.58, 13.22, 11.12, 8.66, 10.2]
vector3 = [9.30, 2.29, 6.63, 6.24, 5.73, 6.36]
vector_new = np.asarray(vector2)
vector_new2 = np.asarray(vector3)
matrix1 = np.array([vector_new, vector_new2]).T
print(' Mean' + '   ' + 'Std Dev')
print(matrix1)

vector3 = [5, 1,3.3, 1.6, 0.3, 1.0]
vector4 = [50, 12.3, 40, 27.9, 29.8, 28.2]
vector_new3 = np.asarray(vector3)
vector_new4 = np.asarray(vector4)
matrix2 = np.array([vector_new3, vector_new4]).T
print(' Min' + '   ' + 'Max')
print(matrix2)

'Arrays:
'
df1 = pd.DataFrame([['AZ', 1], ['GA', 1], ['MD', 1], ['MN', 4], ['NC', 1], ['NJ', 3], ['TN', 1], ['UT', 2]]
, columns = ['State', 'Count'])
print('Data set array\n-----------------\n', df1)

df2 = pd.DataFrame([['2000', 8600], ['2002', 7700], ['2004', 12600], ['2006', 14800], ['2008', 20000]]
, columns = ['Year', 'Pop'])
print('White Prev - AZ\n-----------------\n', df2)

df3 = pd.DataFrame([['2000', 0], ['2002', 2600], ['2004', 11900], ['2006', 16200], ['2008', 19000]]
, columns = ['Year', 'Pop'])
print('API Prev - AZ\n-----------------\n', df3)


'Transformations:
'
df['male_prev_total'] = df.male_prev*1000
df['female_prev_total'] = df.female_prev*1000
df['white_prev_total'] = df.white_prev*1000
df['black_prev_total'] = df.black_prev*1000
df['hisp_prev_total'] = df.hisp_prev*1000
df['api_prev_total'] = df.api_prev*1000

white_sorted = df.sort_values(['white_prev_total'], ascending=[False])
white_sorted.head(10)

black_sorted = df.sort_values(['black_prev_total'], ascending=[False])
black_sorted.head(10)

hisp_sorted = df.sort_values(['hisp_prev_total'], ascending=[False])
hisp_sorted.head(10)

api_sorted = df.sort_values(['api_prev_total'], ascending=[False])
api_sorted.head(10)

state = df[(df.abbr=='AZ')]
state.head()

'Regression plot:
'
sns.regplot(df.male_prev, y = df.white_prev)
sns.regplot(df.female_prev, y = df.hisp_prev)

'Historgram plot:
'
sns.distplot(df.male_prev)
sns.distplot(df.female_prev)
sns.distplot(df.white_prev)
sns.distplot(df.api_prev)

'Line plots:
'
sns.lineplot(data = df, x = df.year, y = df.male_prev)
sns.lineplot(data = df, x = df.year, y = df.female_prev)
sns.lineplot(data = df, x = df.year, y = df.white_prev)
sns.lineplot(data = df, x = df.year, y = df.black_prev)
sns.lineplot(data = df, x = df.year, y = df.hisp_prev)
sns.lineplot(data = df, x = df.year, y = df.api_prev)

x1 = [2000, 2002, 2004, 2006, 2008]
y1 = [8600, 7700, 12600, 14800, 20700]
plt.plot(x1, y1, label = 'White Prev')

x2 = [2000, 2002, 2004, 2006, 2008]
y2 = [0, 2600, 11900, 16200, 19000]
plt.plot(x2, y2, label = 'API Prev')

x3 = [2000, 2002, 2004, 2006, 2008]
y3 = [7300, 6300, 5600, 12900, 16100]
plt.plot(x3, y3, label = 'Black Prev')

x4 = [2000, 2002, 2004, 2006, 2008]
y4 = [0, 3400, 7000, 8300, 3900]
plt.plot(x4, y4, label = 'Hisp Prev')


plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population by Race/Ethnicity')
plt.legend()
plt.show()

'Bar plot:
'
df[‘API_prev>than_avg’].value_counts().plot(kind=’barh’)
