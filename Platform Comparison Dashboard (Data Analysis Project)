import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("AI_Generated_Art_Popularity.csv")

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Average price per platform
#estimator='mean': Calculates the average price for each platform.
#hue="Platform": Forces seaborn to assign different colors per bar.

plt.figure(figsize=(15, 6))
sns.barplot(data=df, x="Platform", y="Price (USD)", estimator='mean', errorbar=None,
            hue="Platform", legend=True) 
plt.title("Average Price per Platform")
plt.ylabel("Average Price (USD)")
plt.tight_layout()
plt.show()


# 2. Average engagement score per platform
plt.figure(figsize=(15, 6))
sns.barplot(data=df, x="Platform", y="Engagement_Score", estimator='mean', errorbar=None,
            hue="Platform", legend =True)  
plt.title("Average Engagement Score per Platform")
plt.ylabel("Average Engagement Score")
plt.tight_layout()
plt.show()

# 3. Creator-type popularity per platform (count of artworks)
plt.figure(figsize=(15, 6))
sns.countplot(data=df, x="Platform", hue="Creator_Type", palette="Set2")
plt.title("Creator-Type Popularity per Platform")
plt.ylabel("Number of Artworks")
plt.tight_layout()
plt.show()
