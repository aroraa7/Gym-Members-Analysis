# **Fitness Data Analysis: Logistic Regression**

## **Overview**
This project analyzes a dataset of gym members' demographic, fitness, and health metrics to explore relationships between variables like calorie burn, workout type, and gender. Using **logistic regression**, we investigated how calorie burn influences gender and workout preferences.

---

## **Logistic Regression Analysis**
### 1. **Gender Prediction**
- **Goal**: Investigate whether **Calories_Burned_Per_Hour** predicts Gender (Male = 1, Female = 0).
- **Findings**:
  - The model showed a statistically significant relationship.
  - Odds of being Male increased slightly (by ~1.1%) for every additional calorie burned per hour.
  - However, the effect was small, and **Calories_Burned_Per_Hour** alone doesn’t explain much of the variation in Gender.
  - Adding features like **experience level** or **workout frequency** could enhance the model’s predictive power.

---

### 2. **Workout Type Analysis**
- **Goal**: Explore whether **Calories_Burned_Per_Hour** predicts workout type (Cardio, HIIT, Strength, Yoga).
- **Findings**:
  - No significant relationship was found between calorie burn and workout type.
  - Coefficients were small, and p-values were insignificant across all workout categories.
  - This suggests **Calories_Burned_Per_Hour** isn’t a key factor in determining workout preferences.
  - Other factors like **experience level**, **session duration**, or **personal goals** might play a larger role.

---

## **Next Possible Steps**
- Include additional predictors like **experience level**, **workout frequency**, and **heart rate metrics** to improve the models.
- Explore **interaction effects** (e.g., how session duration and calorie burn together impact outcomes).
- Try non-linear models like **decision trees** or **random forests** for more complex relationships.
- Test the model on external datasets to evaluate how well it generalizes.

---

## **Dataset Information**
The dataset was retrieved from kaggle, refer to link (here[https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset]), and it includes:
- **Demographics**: Age, Gender, BMI
- **Fitness Metrics**: Calories burned, Session duration, Heart rate (Avg_BPM, Max_BPM, Resting_BPM)
- **Health Metrics**: Fat percentage, Water intake
- **Workout Patterns**: Frequency, Type (Cardio, HIIT, Strength, Yoga), Experience level
