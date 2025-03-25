import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
# https://www.kaggle.com/datasets/atharvasoundankar/viral-social-media-trends-and-engagement-analysis/data
data = pd.read_csv('Viral_Social_Media_Trends.csv')

# Display basic information about the dataset
print("Data Information:")
print(data.info())

print("\nData Description:")
print(data.describe())

# Print the column names to verify
print("\nColumn Names:")
print(data.columns)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Data Visualization

# Distribution of Views, Likes, Shares, and Comments
plt.figure(figsize=(10, 6))
sns.histplot(data['Views'], bins=30, kde=True)
plt.title('Distribution of Views')
plt.xlabel('Views')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Likes'], bins=30, kde=True)
plt.title('Distribution of Likes')
plt.xlabel('Likes')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Shares'], bins=30, kde=True)
plt.title('Distribution of Shares')
plt.xlabel('Shares')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Comments'], bins=30, kde=True)
plt.title('Distribution of Comments')
plt.xlabel('Comments')
plt.ylabel('Frequency')
plt.show()

# Relationship between Views and Likes, Shares, Comments
plt.figure(figsize=(10, 6))
sns.regplot(x='Views', y='Likes', data=data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.title('Relationship between Views and Likes')
plt.xlabel('Views')
plt.ylabel('Likes')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='Views', y='Shares', data=data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.title('Relationship between Views and Shares')
plt.xlabel('Views')
plt.ylabel('Shares')
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='Views', y='Comments', data=data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.title('Relationship between Views and Comments')
plt.xlabel('Views')
plt.ylabel('Comments')
plt.show()

# Count plot of Content_Type
plt.figure(figsize=(10, 6))
sns.countplot(x='Content_Type', data=data)
plt.title('Count of Posts by Content Type')
plt.xlabel('Content Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Count plot of Engagement_Level
plt.figure(figsize=(10, 6))
sns.countplot(x='Engagement_Level', data=data)
plt.title('Count of Posts by Engagement Level')
plt.xlabel('Engagement Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Distribution of Engagement_Level across different Platforms
plt.figure(figsize=(10, 6))
sns.countplot(x='Platform', hue='Engagement_Level', data=data)
plt.title('Distribution of Engagement Level across Platforms')
plt.xlabel('Platform')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(12, 8))
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()