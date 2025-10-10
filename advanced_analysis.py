"""
Advanced Analysis: Additional Machine Learning Techniques
Building on the capstone project with advanced methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" ADVANCED ANALYSIS: Additional ML Techniques")
print("="*80)

# Load data
df = pd.read_csv('day.csv')
df['dteday'] = pd.to_datetime(df['dteday'])

# ========================================
# ANALYSIS 1: USER SEGMENTATION
# ========================================
print("\n" + "="*80)
print("ANALYSIS 1: CASUAL VS REGISTERED USER PATTERNS")
print("="*80)

# Compare casual vs registered users
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribution comparison
axes[0, 0].hist(df['casual'], bins=50, alpha=0.7, label='Casual', edgecolor='black')
axes[0, 0].hist(df['registered'], bins=50, alpha=0.7, label='Registered', edgecolor='black')
axes[0, 0].set_title('Distribution: Casual vs Registered Users', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Number of Rentals')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Temperature impact on user types
temp_bins = pd.cut(df['temp'], bins=10)
casual_by_temp = df.groupby(temp_bins)['casual'].mean()
registered_by_temp = df.groupby(temp_bins)['registered'].mean()

x_labels = [f"{interval.left:.2f}-{interval.right:.2f}" for interval in casual_by_temp.index]
x_pos = np.arange(len(x_labels))

axes[0, 1].plot(x_pos, casual_by_temp.values, marker='o', linewidth=2, label='Casual', markersize=8)
axes[0, 1].plot(x_pos, registered_by_temp.values, marker='s', linewidth=2, label='Registered', markersize=8)
axes[0, 1].set_title('User Types vs Temperature', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Temperature Range')
axes[0, 1].set_ylabel('Average Rentals')
axes[0, 1].set_xticks(x_pos[::2])
axes[0, 1].set_xticklabels(x_labels[::2], rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Weekday patterns by user type
weekday_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
casual_by_weekday = df.groupby('weekday')['casual'].mean()
registered_by_weekday = df.groupby('weekday')['registered'].mean()

x_pos = np.arange(len(weekday_labels))
width = 0.35

axes[1, 0].bar(x_pos - width/2, casual_by_weekday.values, width, label='Casual', alpha=0.8, edgecolor='black')
axes[1, 0].bar(x_pos + width/2, registered_by_weekday.values, width, label='Registered', alpha=0.8, edgecolor='black')
axes[1, 0].set_title('User Types by Weekday', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Day of Week')
axes[1, 0].set_ylabel('Average Rentals')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(weekday_labels)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# User ratio analysis
df['casual_ratio'] = df['casual'] / df['cnt']
df['registered_ratio'] = df['registered'] / df['cnt']

axes[1, 1].scatter(df['temp'], df['casual_ratio'], alpha=0.5, s=30, edgecolor='k', label='Casual %')
axes[1, 1].set_title('Casual User Percentage vs Temperature', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Temperature (normalized)')
axes[1, 1].set_ylabel('Casual User Ratio')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('visualizations/10_user_segmentation.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: visualizations/10_user_segmentation.png")
print(f"\nKey Insights:")
print(f"  Average Casual Users: {df['casual'].mean():.0f} per day")
print(f"  Average Registered Users: {df['registered'].mean():.0f} per day")
print(f"  Registered to Casual Ratio: {df['registered'].mean() / df['casual'].mean():.2f}:1")
print(f"  Weekend Casual Boost: {df[df['weekday'].isin([0,6])]['casual'].mean() / df[df['weekday'].isin([1,2,3,4,5])]['casual'].mean():.2f}x")

# ========================================
# ANALYSIS 2: CLUSTERING ANALYSIS
# ========================================
print("\n" + "="*80)
print("ANALYSIS 2: K-MEANS CLUSTERING - IDENTIFYING RENTAL PATTERNS")
print("="*80)

# Prepare features for clustering
cluster_features = ['temp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
X_cluster = df[cluster_features].copy()

# Standardize features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Elbow method to find optimal clusters
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster_scaled)
    inertias.append(kmeans.inertia_)

# Perform clustering with optimal k
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_cluster_scaled)

# Visualize clusters
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Elbow curve
axes[0, 0].plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
axes[0, 0].set_title('Elbow Method - Optimal Clusters', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Clusters in temp vs rentals
colors = ['red', 'blue', 'green', 'orange']
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    axes[0, 1].scatter(cluster_data['temp'], cluster_data['cnt'], 
                       c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6, s=50, edgecolor='k')
axes[0, 1].set_title('Clusters: Temperature vs Rentals', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Temperature (normalized)')
axes[0, 1].set_ylabel('Total Rentals')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Cluster characteristics
cluster_summary = df.groupby('cluster')[['temp', 'hum', 'windspeed', 'cnt']].mean()
cluster_summary_scaled = (cluster_summary - cluster_summary.mean()) / cluster_summary.std()

axes[1, 0].matshow(cluster_summary_scaled.T, cmap='RdYlGn', aspect='auto')
axes[1, 0].set_xticks(range(optimal_k))
axes[1, 0].set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
axes[1, 0].set_yticks(range(len(cluster_summary.columns)))
axes[1, 0].set_yticklabels(cluster_summary.columns)
axes[1, 0].set_title('Cluster Characteristics (Standardized)', fontweight='bold', fontsize=12, pad=20)

# Add values
for i in range(len(cluster_summary.columns)):
    for j in range(optimal_k):
        text = axes[1, 0].text(j, i, f'{cluster_summary_scaled.iloc[j, i]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')

# Cluster size distribution
cluster_counts = df['cluster'].value_counts().sort_index()
axes[1, 1].bar(range(optimal_k), cluster_counts.values, edgecolor='black', alpha=0.7, color=colors)
axes[1, 1].set_title('Cluster Size Distribution', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Number of Days')
axes[1, 1].set_xticks(range(optimal_k))
axes[1, 1].set_xticklabels([f'Cluster {i}' for i in range(optimal_k)])
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate(cluster_counts.values):
    axes[1, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('visualizations/11_clustering_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: visualizations/11_clustering_analysis.png")
print(f"\nCluster Insights:")
for cluster in range(optimal_k):
    cluster_data = df[df['cluster'] == cluster]
    print(f"\n  Cluster {cluster}:")
    print(f"    Size: {len(cluster_data)} days ({len(cluster_data)/len(df)*100:.1f}%)")
    print(f"    Avg Rentals: {cluster_data['cnt'].mean():.0f}")
    print(f"    Avg Temp: {cluster_data['temp'].mean():.3f}")
    print(f"    Avg Humidity: {cluster_data['hum'].mean():.3f}")

# ========================================
# ANALYSIS 3: NEURAL NETWORK MODEL
# ========================================
print("\n" + "="*80)
print("ANALYSIS 3: NEURAL NETWORK REGRESSION")
print("="*80)

# Prepare data
feature_cols = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
X = df[feature_cols]
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features for neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train neural network
print("\nTraining Neural Network...")
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), 
                        max_iter=1000, 
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.2)
nn_model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_pred_train = nn_model.predict(X_train_scaled)
y_pred_test = nn_model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"✓ Neural Network Trained")
print(f"  Architecture: 11 → 100 → 50 → 25 → 1")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²: {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.2f}")
print(f"  Test MAE: {test_mae:.2f}")

# Visualize neural network performance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Predictions vs Actual
axes[0].scatter(y_test, y_pred_test, alpha=0.5, s=50, edgecolor='k')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Bike Rentals', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Predicted Bike Rentals', fontweight='bold', fontsize=12)
axes[0].set_title(f'Neural Network Predictions (R²={test_r2:.4f})', fontweight='bold', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning curve (loss over iterations)
axes[1].plot(nn_model.loss_curve_, linewidth=2)
axes[1].set_xlabel('Iteration', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Loss', fontweight='bold', fontsize=12)
axes[1].set_title('Neural Network Learning Curve', fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/12_neural_network.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: visualizations/12_neural_network.png")

# ========================================
# ANALYSIS 4: PRINCIPAL COMPONENT ANALYSIS
# ========================================
print("\n" + "="*80)
print("ANALYSIS 4: DIMENSIONALITY REDUCTION - PCA")
print("="*80)

# Perform PCA
pca_features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
X_pca = df[pca_features]
X_pca_scaled = StandardScaler().fit_transform(X_pca)

pca = PCA()
X_pca_transformed = pca.fit_transform(X_pca_scaled)

# Visualize PCA
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

axes[0, 0].bar(range(1, len(explained_var)+1), explained_var, alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Principal Component', fontweight='bold')
axes[0, 0].set_ylabel('Explained Variance Ratio', fontweight='bold')
axes[0, 0].set_title('PCA - Explained Variance by Component', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Cumulative explained variance
axes[0, 1].plot(range(1, len(cumulative_var)+1), cumulative_var, marker='o', linewidth=2, markersize=8)
axes[0, 1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Variance')
axes[0, 1].set_xlabel('Number of Components', fontweight='bold')
axes[0, 1].set_ylabel('Cumulative Explained Variance', fontweight='bold')
axes[0, 1].set_title('Cumulative Explained Variance', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# PC1 vs PC2 colored by rentals
scatter = axes[1, 0].scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1],
                             c=df['cnt'], cmap='viridis', s=50, edgecolor='k', alpha=0.7)
axes[1, 0].set_xlabel('First Principal Component', fontweight='bold')
axes[1, 0].set_ylabel('Second Principal Component', fontweight='bold')
axes[1, 0].set_title('PCA: First Two Components (colored by rentals)', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 0], label='Bike Rentals')

# Feature contributions to PC1
pc1_contributions = pd.DataFrame({
    'feature': pca_features,
    'contribution': np.abs(pca.components_[0])
}).sort_values('contribution', ascending=True)

axes[1, 1].barh(pc1_contributions['feature'], pc1_contributions['contribution'], edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Absolute Contribution', fontweight='bold')
axes[1, 1].set_title('Feature Contributions to First Principal Component', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('visualizations/13_pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: visualizations/13_pca_analysis.png")
print(f"\nPCA Insights:")
print(f"  First 2 components explain: {cumulative_var[1]*100:.2f}% of variance")
print(f"  First 5 components explain: {cumulative_var[4]*100:.2f}% of variance")
print(f"  Components for 95% variance: {np.argmax(cumulative_var >= 0.95) + 1}")

# ========================================
# ANALYSIS 5: ADVANCED TIME SERIES PATTERNS
# ========================================
print("\n" + "="*80)
print("ANALYSIS 5: ADVANCED TIME SERIES ANALYSIS")
print("="*80)

# Rolling statistics
df['cnt_rolling_7'] = df['cnt'].rolling(window=7).mean()
df['cnt_rolling_30'] = df['cnt'].rolling(window=30).mean()
df['cnt_rolling_std_7'] = df['cnt'].rolling(window=7).std()

# Monthly growth analysis
df['year_month'] = df['dteday'].dt.to_period('M')
monthly_stats = df.groupby('year_month').agg({
    'cnt': ['mean', 'sum', 'std']
}).reset_index()
monthly_stats.columns = ['year_month', 'mean', 'sum', 'std']
monthly_stats['year_month_str'] = monthly_stats['year_month'].astype(str)

# Visualize advanced time series
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# Rolling averages
axes[0].plot(df['dteday'], df['cnt'], alpha=0.3, label='Daily', linewidth=1)
axes[0].plot(df['dteday'], df['cnt_rolling_7'], linewidth=2, label='7-day Moving Avg')
axes[0].plot(df['dteday'], df['cnt_rolling_30'], linewidth=2, label='30-day Moving Avg')
axes[0].fill_between(df['dteday'], 
                     df['cnt_rolling_7'] - df['cnt_rolling_std_7'],
                     df['cnt_rolling_7'] + df['cnt_rolling_std_7'],
                     alpha=0.2, label='±1 Std Dev (7-day)')
axes[0].set_title('Bike Rentals with Moving Averages', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Bike Rentals')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Monthly trends
x_pos = np.arange(len(monthly_stats))
axes[1].bar(x_pos, monthly_stats['sum'], edgecolor='black', alpha=0.7)
axes[1].set_title('Monthly Total Rentals', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Total Rentals')
axes[1].set_xticks(x_pos[::2])
axes[1].set_xticklabels(monthly_stats['year_month_str'].iloc[::2], rotation=45, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')

# Month-over-month growth
monthly_stats['growth'] = monthly_stats['sum'].pct_change() * 100
axes[2].plot(x_pos[1:], monthly_stats['growth'].iloc[1:], marker='o', linewidth=2, markersize=8, color='green')
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[2].fill_between(x_pos[1:], 0, monthly_stats['growth'].iloc[1:], 
                     where=(monthly_stats['growth'].iloc[1:] > 0), alpha=0.3, color='green', label='Growth')
axes[2].fill_between(x_pos[1:], 0, monthly_stats['growth'].iloc[1:],
                     where=(monthly_stats['growth'].iloc[1:] < 0), alpha=0.3, color='red', label='Decline')
axes[2].set_title('Month-over-Month Growth Rate', fontweight='bold', fontsize=12)
axes[2].set_xlabel('Month')
axes[2].set_ylabel('Growth Rate (%)')
axes[2].set_xticks(x_pos[1::2])
axes[2].set_xticklabels(monthly_stats['year_month_str'].iloc[1::2], rotation=45, ha='right')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/14_advanced_time_series.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: visualizations/14_advanced_time_series.png")

# Calculate year-over-year growth
y2011_avg = df[df['yr'] == 0]['cnt'].mean()
y2012_avg = df[df['yr'] == 1]['cnt'].mean()
yoy_growth = ((y2012_avg - y2011_avg) / y2011_avg) * 100

print(f"\nTime Series Insights:")
print(f"  2011 Average Daily Rentals: {y2011_avg:.0f}")
print(f"  2012 Average Daily Rentals: {y2012_avg:.0f}")
print(f"  Year-over-Year Growth: {yoy_growth:.1f}%")
print(f"  Peak Month: {monthly_stats.loc[monthly_stats['sum'].idxmax(), 'year_month_str']}")
print(f"  Peak Monthly Rentals: {monthly_stats['sum'].max():.0f}")

# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "="*80)
print(" ADVANCED ANALYSIS COMPLETE")
print("="*80)

print("\n✓ 5 Advanced Analyses Completed:")
print("  1. User Segmentation (Casual vs Registered)")
print("  2. K-Means Clustering (4 clusters identified)")
print("  3. Neural Network Regression")
print("  4. Principal Component Analysis (PCA)")
print("  5. Advanced Time Series Analysis")

print("\n✓ 5 New Visualizations Created:")
print("  10_user_segmentation.png")
print("  11_clustering_analysis.png")
print("  12_neural_network.png")
print("  13_pca_analysis.png")
print("  14_advanced_time_series.png")

print("\n" + "="*80)
print(" ALL ANALYSES SAVED TO visualizations/ FOLDER")
print("="*80)
print()
