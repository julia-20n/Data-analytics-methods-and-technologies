import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv("/content/drive/My Drive/cleaned_usa_rain_prediction_dataset_2024_2025.csv")

numeric_features = data[['Temperature', 'Humidity', 'Wind Speed', 'Precipitation', 'Cloud Cover', 'Pressure']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)
sampled_features = scaled_features[:20000]

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(sampled_features)

cluster_labels = kmeans.labels_
X_train, X_test, y_train, y_test = train_test_split(sampled_features, cluster_labels, test_size=0.3, random_state=42)

accuracies = []
k_values = range(1, 21)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracies, marker='o', label='Точність')
plt.axhline(y=0.85, color='r', linestyle='--', label='85% точності')
plt.xlabel('Кількість сусідів (k)')
plt.ylabel('Точність')
plt.title('Вплив кількості сусідів на точність')
plt.legend()
plt.grid()
plt.savefig("knn_accuracy_plot.png")
plt.show()

optimal_k = k_values[np.argmax([acc for acc in accuracies if acc >= 0.85])]
print(f"Оптимальне k для точності ≥ 85%: {optimal_k}")

knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)
y_pred_test = knn_optimal.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Точність на тестовій вибірці (k={optimal_k}): {test_accuracy:.2%}")

cross_val_accuracies = cross_val_score(knn_optimal, sampled_features, cluster_labels, cv=3, scoring='accuracy')
print(f"Точність крос-валідації (3 блоки): {cross_val_accuracies.mean():.2%}")

