import pandas as pd
df = pd.read_csv('./dataset.csv')

# Stat Descriptions
df.head()
df.describe(include = 'all')
print(df.isnull().sum())
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots(2, 1, figsize=(15, 10))
ax[0].hist(df['popularity'], bins=30, edgecolor='black', alpha=0.7)
ax[0].set_title('Histogram of Popularity')
ax[0].set_xlabel('Popularity')
ax[0].set_ylabel('Frequency')
from scipy.stats import gaussian_kde
density = gaussian_kde(df['popularity'])
x_vals = np.linspace(min(df['popularity']), max(df['popularity']), 1000)
y_vals = density(x_vals)
ax[1].plot(x_vals, y_vals, color='blue')
ax[1].set_title('Density Plot of Popularity')
ax[1].set_xlabel('Popularity')
ax[1].set_ylabel('Density')
plt.tight_layout()
plt.show()
popularity_stats = df['popularity'].describe()
print(popularity_stats)
plt.figure(figsize=(10, 5))
plt.boxplot(df['popularity'], vert=False)
plt.title('Box Plot of Popularity')
plt.xlabel('Popularity')
plt.show()
numerical_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numerical_df.corr()
correlation_with_popularity = correlation_matrix['popularity'].sort_values(ascending=False)
correlation_with_popularity