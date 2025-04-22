import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore  # <-- for Z-score

# Load the dataset
df = pd.read_csv("AI_Generated_Art_Popularity.csv")

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Average price per platform
avg_price = df.groupby("Platform")["Price"].mean().sort_values(ascending=False).reset_index()

# 2. Average engagement score per platform
avg_engagement = df.groupby("Platform")["Engagement_Score"].mean().sort_values(ascending=False).reset_index()

# 3. Creator-type popularity per platform
creator_popularity = df.groupby(["Platform", "Creator_Type"]).size().reset_index(name="Count")

print(avg_price)
print(avg_engagement)
print(creator_popularity)


#-----------------------------------------------------------------------------------

# Plotting all three aspects
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

#Creates 3 vertical subplots, big enough for detailed visual comparison.


# Avg Price Plot
sns.barplot(data=avg_price, x="Platform", y="Price", hue="Platform", dodge=False, ax=axs[0], palette="viridis")
axs[0].set_title("Average Price per Platform")
axs[0].set_ylabel("Average Price")
axs[0].set_xlabel("Platform")


# Avg Engagement Plot
sns.barplot(data=avg_engagement, x="Platform", y="Engagement_Score", hue="Platform", dodge=False, ax=axs[1], palette="magma")
axs[1].set_title("Average Engagement Score per Platform")
axs[1].set_ylabel("Average Engagement Score")
axs[1].set_xlabel("Platform")


# Creator-Type Popularity Plot
sns.barplot(data=creator_popularity, x="Platform", y="Count", hue="Creator_Type", ax=axs[2])
axs[2].set_title("Creator-Type Popularity per Platform")
axs[2].set_ylabel("Count")
axs[2].set_xlabel("Platform")

plt.tight_layout()
plt.show()


#-----------------------------------------------------------------------------------

# Aggregate metrics per platform
platform_stats = df.groupby("Platform").agg({
    "Price": "mean",
    "Engagement_Score": "mean",
    "Creator_Type": lambda x: x.value_counts().max()
}).rename(columns={
    "Price": "Avg_Price",
    "Engagement_Score": "Avg_Engagement",
    "Creator_Type": "Top_Creator_Count"
})

# Normalize the metrics using Z-score
zscore_stats = platform_stats.apply(zscore)

# Compute composite score
zscore_stats["Composite_Score"] = zscore_stats.mean(axis=1)

# Identify best platform
best_platform = zscore_stats["Composite_Score"].idxmax()
best_score = zscore_stats.loc[best_platform, "Composite_Score"]

print(f"✅ Best Performing Platform (Z-score method): {best_platform} (Score: {best_score:.2f})")

#-----------------------------------------------------------------------------------


plt.figure(figsize=(10, 6))
sns.heatmap(zscore_stats.sort_values("Composite_Score", ascending=False), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Platform Performance (Z-score Normalized Metrics)", fontsize=14)
plt.xlabel("Metric")
plt.ylabel("Platform")
plt.tight_layout()
plt.show()
