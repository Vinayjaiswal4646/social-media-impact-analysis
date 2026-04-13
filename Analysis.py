import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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