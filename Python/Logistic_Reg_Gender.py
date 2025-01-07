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


# Perform Logistic Reg on binaray predictor; Gender
genderdf = caloriesdf.copy()
# Change Gender as a binary variable, Female = 0, Male = 1
# genderdf['Gender_Con'] = genderdf['Gender'].replace({'Male': 1, 'Female': 0})
gender_mapping = {'Male': 1, 'Female': 0}
genderdf['Gender_Con'] = genderdf['Gender'].map(gender_mapping)
# display(genderdf.head())

# Define predictor (Calories_Burned_Per_Hour) and response (Gender_con)
X = genderdf[['Calories_Burned_Per_Hour']]
Y = genderdf['Gender_Con']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Add an intercept to the training predictors (for statsmodels)
X_train_sm = sm.add_constant(X_train)

# Fit logistic regression using statsmodels
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

# Display the summary
print(result.summary())

# Extract coefficients, standard errors, and p-values
coefficients = result.params
standard_errors = result.bse
p_values = result.pvalues

# Print extracted results
print()
print("Coefficients:\n", coefficients)
print("Standard Errors:\n", standard_errors)
print("P-values:\n", p_values)