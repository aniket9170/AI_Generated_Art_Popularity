import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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

# Plotting all three aspects
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

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

# Normalize the metrics using MinMaxScaler
scaler = MinMaxScaler()
normalized_stats = pd.DataFrame(
    scaler.fit_transform(platform_stats),
    columns=platform_stats.columns,
    index=platform_stats.index
)

# Compute composite score
normalized_stats["Composite_Score"] = normalized_stats.mean(axis=1)

# Identify best platform
best_platform = normalized_stats["Composite_Score"].idxmax()
best_score = normalized_stats.loc[best_platform, "Composite_Score"]

print(f"✅ Best Performing Platform: {best_platform} (Score: {best_score:.2f})")


