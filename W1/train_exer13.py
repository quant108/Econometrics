import statistics as stats
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np

df = pd.read_excel('TrainExer13.xls')

print("Column headings:")
print(df)

data = df.values

y = np.array(df['Winning time men'])
x = np.array(df['Game'])
n = len(x)

y1 = y - np.mean(y)
x1 = x - np.mean(x)

b = np.inner(y1, x1)/np.inner(x1, x1)
a = np.mean(y) - b*np.mean(x)

e = y - a - b*x
s2 = np.inner(e, e)/(n-2)
s = np.sqrt(s2)
R2 = 1 - np.inner(e, e)/np.inner(y1, y1)

print(a, b, R2, s)
xi = np.array([16, 17, 18])
yi = a + b*xi
print(yi)

