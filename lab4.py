import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')
data = pd.read_csv("/content/drive/My Drive/cleaned_usa_rain_prediction_dataset_2024_2025.csv")

selected_features = data[['Wind Speed', 'Humidity']]
scaler = StandardScaler()
scaled_selected_features = scaler.fit_transform(selected_features)
sampled_features = scaled_selected_features[:20000]

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(sampled_features)
cluster_labels = kmeans.labels_
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()

largest_clusters = cluster_counts.nlargest(2).index
selected_data = sampled_features[np.isin(cluster_labels, largest_clusters)]
selected_labels = np.where(np.isin(cluster_labels, largest_clusters), cluster_labels, -1)
selected_labels = selected_labels[selected_labels != -1]

class_mapping = {largest_clusters[0]: 0, largest_clusters[1]: 1}
selected_labels = np.vectorize(class_mapping.get)(selected_labels)
X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_labels, test_size=0.3, random_state=42)

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точність класифікації методом SVM з лінійним ядром: {accuracy}")

plt.figure(figsize=(10, 7))
sns.scatterplot(x=selected_data[:, 0], y=selected_data[:, 1], hue=selected_labels, palette='Set1', s=20, legend="full")
plt.xlabel('Температура')
plt.ylabel('Вологість')
plt.title(f'Класифікація SVM (лінійне ядро) для двох найбільших кластерів')

w = svm.coef_[0]
b = svm.intercept_[0]
x_vals = np.linspace(selected_data[:, 0].min(), selected_data[:, 0].max(), 100)
y_vals = -(w[0] / w[1]) * x_vals - b / w[1]
plt.plot(x_vals, y_vals, 'k--', label='Гіперплощина розділення')
plt.legend()
plt.show()
