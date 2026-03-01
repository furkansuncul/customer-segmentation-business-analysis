import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Struct
if not os.path.exists('visuals'):
    os.makedirs('visuals')

# Upload the data
df = pd.read_csv("Mall_Customers.csv")

# Annual Income and spend score
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method (Optimal Cluster Selection)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig("visuals/elbow_method.png")
plt.close()

# Final Model & Clustering (Selected n=5)
# Note: I chose 5 clusters because n=3 misses the "Target" group
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Silhouette score
score = silhouette_score(X_scaled, kmeans.labels_)
print(f"✅ Model Silhouette Score: {round(score, 3)}")

# Segment names
cluster_names = {
    0: "Target (High Income, Low Spend)",
    1: "Standard (Mid Income, Mid Spend)",
    2: "Sensible (Low Income, Low Spend)",
    3: "Careless (Low Income, High Spend)",
    4: "Champions (High Income, High Spend)"
}
df["Segment"] = df["Cluster"].map(cluster_names)

# Visualisation
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Segment",
    palette="viridis",
    data=df,
    s=100,
    edgecolor='black'
)
plt.title('Customer Segments: Income vs Spending', fontsize=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("visuals/customer_segments.png")
plt.show()

# Predict 
# spend_score * 10 (random number)
df['Estimated_CLV'] = df['Spending Score (1-100)'] * 100 

print("\n🚀 --- Summary ---")
summary = df.groupby("Segment").agg({
    'Annual Income (k$)': 'mean',
    'Spending Score (1-100)': 'mean',
    'Estimated_CLV': 'sum',
    'Cluster': 'count'
}).rename(columns={'Cluster': 'Customer_Count'})

print(summary)

# Save
summary.to_csv("segmentation_report.csv")


def predict_segment(income, spend_score):
    """
    Predicts the strategic segment for a new customer.    
    """
    
    # Passing data as DataFrame to maintain feature names and avoid UserWarning.
    yeni_veri_df = pd.DataFrame([[income, spend_score]], 
                                columns=["Annual Income (k$)", "Spending Score (1-100)"])
    
    yeni_veri_scaled = scaler.transform(yeni_veri_df)
    
    kume = kmeans.predict(yeni_veri_scaled)[0]
    segment_name = cluster_names[kume]
    
    print(f"💰 Income: {income}, Score: {spend_score} -> SEGMENT: {segment_name}")

    
    if "Champions" in segment_name:
        return "Action: High-value customer. Enroll in loyalty programs and VIP offers."
    elif "Target" in segment_name:
        return "Action: High potential. Needs personalized marketing to increase spending."
    elif "Careless" in segment_name:
        return "Action: High spender but low income. Offer flexible payment options."
    else:
        return "Action: Standard monitoring."
    
    
print(predict_segment(100, 20))