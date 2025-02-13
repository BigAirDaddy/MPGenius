import pandas as pd
import numpy as np


column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year', 'Origin','Car Name' ]

file_path = 'auto-mpg.data'

df = pd.read_csv(file_path, names=column_names, delim_whitespace=True)

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df['Horsepower'] = df['Horsepower'].astype(float)
df['Model Year'] = df['Model Year'].apply(lambda x: x + 1900)
cleaned_file_path = 'Data/auto.csv'

df.to_csv(cleaned_file_path, index=False)

print(df.head())
