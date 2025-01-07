import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


# Reading in and looking at dataset
df = pd.read_csv('gym_members_exercise_tracking.csv')
print(df.head(10))

# Look at the stats and data types within each feature
df.describe()
df.info()

#created a function that converts the inches to feet and inches for height
def inches_to_feet_inches(inches):
    feet = int(inches // 12)
    remaining_inches = int(inches % 12)
    return float(f"{feet}.{remaining_inches}")

dfcopy = df.copy()

#created new columns for weight in lbs and height inches
dfcopy['Weight_lbs'] = round(dfcopy['Weight (kg)'] * 2.20462, 2)

dfcopy['Height_inches'] = round(dfcopy['Height (m)'] * 39.3701, 2)

#used the function on the new col for height in inches
dfcopy['Height_ft'] = dfcopy['Height_inches'].apply(inches_to_feet_inches)

#drop the cols not needed anymore
dfcopy = dfcopy.drop(['Weight (kg)', 'Height (m)', 'Height_inches'], axis=1)

print(dfcopy.head(10))
print(dfcopy.info())

#checking to see of there are any null values in the dataframe
any_null = dfcopy.isnull().values.any()
print(f"Any null values in the Data: {any_null}")

#double checking by counting
null_counts = dfcopy.isnull().sum()
print(f"\nNull value count in each column:")
print(null_counts)









