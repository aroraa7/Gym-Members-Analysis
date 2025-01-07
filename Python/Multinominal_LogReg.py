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


# Reading in cleaned data
df = pd.read_csv('gymmembers_exercise_New.csv')


# Calculating a response variable:
# Calculate Calories_Burned_Per_Hour using this formula
# Calories Burned Per Hour= Calories Burned/Session Duration(hours)

#create a copy so it doesn't ruin prior copy
caloriesdf = df.copy()

# Add a new column for Calories Burned Per Hour
caloriesdf['Calories_Burned_Per_Hour'] = caloriesdf['Calories_Burned'] / caloriesdf['Session_Duration (hours)']


print(f"New Dataframe with Response Variable; Calories_Burned_Per_Hour:\n")
print(caloriesdf)


# Make a copy to keep data from messing up
workoutdf = caloriesdf.copy()


# Change  Workout_Type as a numerical variable
workout_encoder = LabelEncoder()
workoutdf['Workout_Type_Encoded'] = workout_encoder.fit_transform(workoutdf['Workout_Type'])

print(f'Cardio = 0, HIIT = 1, Strength = 2, Yoga = 3')
print()

# Define features (Calories_Burned_Per_Hour) and response (Workout_Type_Encoded)
X = workoutdf[['Calories_Burned_Per_Hour']]
y = workoutdf['Workout_Type_Encoded']

# Add a constant to the predictors
X = sm.add_constant(X)

# Fit multinomial logistic regression model
mnlogit_model = sm.MNLogit(y, X)
result = mnlogit_model.fit()

# Display the summary of the model
print(result.summary())
print()

# Extract coefficients, standard errors, and p-values
coefficients = result.params
standard_errors = result.bse
p_values = result.pvalues

# Display extracted results
print("Coefficients:\n", coefficients)
print("Standard Errors:\n", standard_errors)
print("P-values:\n", p_values)