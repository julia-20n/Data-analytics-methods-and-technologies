import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from google.colab import drive

drive.mount('/content/drive')
data = pd.read_csv("/content/drive/My Drive/cleaned_usa_rain_prediction_dataset_2024_2025.csv")

# Вибір двох атрибутів для кластеризації
selected_features = data[['Temperature', 'Humidity']]
scaler = StandardScaler()
scaled_selected_features = scaler.fit_transform(selected_features)

sampled_features = scaled_selected_features[:20000]

# Метод ліктя
def optimal_k_means(data, max_k=10):
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Кількість кластерів')
    plt.ylabel('SSE (Сума квадратів відхилень)')
    plt.title('Метод ліктя для визначення оптимальної кількості кластерів')
    plt.show()

optimal_k_means(sampled_features)

# Припустимо, оптимальна кількість кластерів — 4
optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(sampled_features)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_labels = kmeans.labels_
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
cluster_summary = pd.DataFrame(cluster_centers, columns=selected_features.columns)
cluster_summary['Розмір кластера'] = cluster_counts.values
cluster_summary.index.name = 'Кластер'
print("Центри кластерів та розміри (K-середніх, 2 атрибути):\n", cluster_summary)

# Візуалізація кластерів для K-середніх
plt.figure(figsize=(10, 7))
sns.scatterplot(x=sampled_features[:, 0], y=sampled_features[:, 1], hue=cluster_labels, palette='Set1', s=20)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', marker='X', s=200, label='Центри')
plt.xlabel('Температура')
plt.ylabel('Вологість')
plt.title(f'Кластеризація методом K-середніх (2 атрибути, k={optimal_k})')
plt.legend()
plt.show()

# Ієрархічна кластеризація
linked = linkage(sampled_features, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=10)
plt.title('Дендрограма для ієрархічної кластеризації (2 атрибути)')
plt.xlabel('Індекс зразка')
plt.ylabel('Відстань')
plt.show()

# Ієрархічна кластеризація для двох атрибутів
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(sampled_features)
hierarchical_cluster_counts = pd.Series(hierarchical_labels).value_counts().sort_index()

hierarchical_summary = pd.DataFrame({
    'Кластер': range(optimal_k),
    'Розмір кластера': hierarchical_cluster_counts.values
})
print("Розміри кластерів (ієрархічний метод, 2 атрибути):\n", hierarchical_summary)

# Розрахунок дисперсії розмірів кластерів для обох методів
k_means_variance = np.var(cluster_counts.values, ddof=1)
hierarchical_variance = np.var(hierarchical_cluster_counts.values, ddof=1)

print("\nДисперсія розмірів кластерів (K-середніх):", k_means_variance)
print("Дисперсія розмірів кластерів (ієрархічний метод):", hierarchical_variance)

if k_means_variance < hierarchical_variance:
    print("\nМетод K-середніх створює більш збалансовані кластери.")
else:
    print("\nІєрархічний метод створює більш збалансовані кластери.")
