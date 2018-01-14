import statistics as stats
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_excel('W1/TrainExer11.xls')

print("Column headings:")
print(df)

Observation = list(df['Observation'])
Age = list(df['Age'])
Expenditures = list(df['Expenditures'])

print([stats.mean(Age), stats.median(Age), stats.stdev(Age), min(Age), max(Age)])
print([stats.mean(Expenditures), stats.median(Expenditures), stats.stdev(Expenditures), min(Expenditures), max(Expenditures)])
