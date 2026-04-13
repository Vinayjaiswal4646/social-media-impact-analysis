import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
#EDA
print("================================================================================================================")


df = pd.read_csv("Social_media_impact_on_std_.csv")
print(df.head())

print("================================================================================================================")
#basic Info
print("Shape:=> ")
print(df.shape)

print("================================================================================================================")

print("Columns ")
print(df.columns)

print("================================================================================================================")

print("Information")
df.info()

print("================================================================================================================")

print("describe")
print(df.describe())

print("================================================================================================================")

print("Null Values")
print(df.isnull().sum())
print("================================================================================================================")

Q1 = df['Avg_Daily_Usage_Hours'].quantile(0.25)
Q3 = df['Avg_Daily_Usage_Hours'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Avg_Daily_Usage_Hours'] < Q1 - 1.5*IQR) |   (df['Avg_Daily_Usage_Hours'] > Q3 + 1.5*IQR)]
print("Outliers",outliers)
#Outliers were analyzed using the IQR method. However, since the detected values were found to be realistic 
# and representative of actual user behavior, they were retained in the dataset to preserve the integrity of the analysis.

#Visualizing outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df[['Avg_Daily_Usage_Hours',  'Sleep_Hours_Per_Night',  'Mental_Health_Score']])
plt.title("Outlier Detection for Key Numerical Features")

plt.show()
#The box plots were used to identify outliers in key numerical variables such as average daily usage, sleep hours,
#and mental health score. A few data points were observed outside the whiskers, indicating the presence of outliers. 
#However, these values appear to be realistic and represent natural variations in user behavior, and hence were retained for further analysis.



#DATA CLEANING DONE


#Objective 1: To analyze the relationship between time spent on social media and sleep duration using data visualization techniques.

#This analysis focuses on examining the relationship between time spent on social media and sleep duration. 
# With increasing use of digital platforms, it is important to understand how social media usage impacts daily lifestyle patterns such as sleep. 
# Visualization techniques are used to identify trends and patterns between these variables.

plt.figure(figsize=(8,5))

sns.scatterplot(x='Avg_Daily_Usage_Hours', y='Sleep_Hours_Per_Night', data=df)
plt.title("Social Media Usage vs Sleep Duration")
plt.xlabel("Average Daily Usage (Hours)")
plt.ylabel("Sleep Hours Per Night")

plt.show()

sns.regplot(x='Avg_Daily_Usage_Hours', y='Sleep_Hours_Per_Night', data=df)

plt.title("Social Media Usage vs Sleep Duration (Regression Trend)")
plt.show()
print("================================================================================================================")

# Objective 2: To analyze the impact of social media usage on academic performance using data visualization techniques.

#This analysis aims to examine whether social media usage has an impact on academic performance.
# By comparing the average daily usage of individuals who report that social media affects their academic performance
# with those who do not, we can identify patterns and differences between the two groups.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))

sns.boxplot(x='Affects_Academic_Performance', y='Avg_Daily_Usage_Hours', data=df)

plt.title("Impact of Social Media Usage on Academic Performance")
plt.xlabel("Affects Academic Performance")
plt.ylabel("Average Daily Usage (Hours)")

plt.show()

#Correlation ;
numeric_df = df.select_dtypes(include=['int64', 'float64'])
corr = numeric_df.corr()
print(corr)

#Heat Map
plt.figure()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Heatmap")
plt.show()


# Objective 3: To analyze the relationship between average daily social media usage and mental health score using Simple Linear Regression.


# This analysis applies Simple Linear Regression to examine the relationship between average daily social media usage and mental health score.
# The objective is to determine whether increased usage has a measurable impact on mental health.
# Regression analysis helps in identifying trends and predicting the dependent variable based on the independent variable.

X = df[['Avg_Daily_Usage_Hours']]
y = df['Mental_Health_Score']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)


plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, color='red', label="Regression Line")

plt.title("Social Media Usage vs Mental Health Score")
plt.xlabel("Avg Daily Usage (Hours)")
plt.ylabel("Mental Health Score")
plt.legend()
plt.show()

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)


# Objective 4:
# To test whether there is a significant difference in average daily social media usage between students whose academic performance is affected and those whose performance is not affected.

# H0 (Null Hypothesis): There is no significant difference in average daily social media usage between students
#  whose academic performance is affected and those whose performance is not affected.
# H1 (Alternative Hypothesis): There is a significant difference in average daily social media usage between the two groups.


group_yes = df[df['Affects_Academic_Performance'] == 'Yes']['Avg_Daily_Usage_Hours']
group_no = df[df['Affects_Academic_Performance'] == 'No']['Avg_Daily_Usage_Hours']


t_stat, p_value = ttest_ind(group_yes, group_no)
print("T-statistic:", t_stat)
print("P-value:", p_value)

# The t-test results show that the p-value is significantly less than 0.05, 
# indicating a statistically significant difference between the two groups. Therefore,
#  the null hypothesis is rejected. This implies that there is a significant difference in average daily social media usage between students 
# whose academic performance is affected and those whose performance is not affected.

# The high t-statistic value further supports the strength of this difference, 
# suggesting that students who report an impact on their academic performance tend to have considerably higher social media usage.




# Objective 5:
# To analyze the impact of different social media platforms on mental health score using data visualization techniques.


# This analysis examines how mental health scores vary across different social media platforms. 
# By comparing users based on their most frequently used platform, the aim is to identify whether certain platforms are associated with better
# or worse mental health outcomes.

plt.figure(figsize=(10,6))

sns.boxplot(x='Most_Used_Platform',y='Mental_Health_Score',data=df)
plt.xticks(rotation=45)
plt.title("Mental Health Score by Most Used Platform")
plt.xlabel("Most Used Platform")
plt.ylabel("Mental Health Score")

plt.show()