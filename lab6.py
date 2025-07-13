import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/My Drive/cleaned_usa_rain_prediction_dataset_2024_2025.csv'
data = pd.read_csv(file_path)

numeric_data = data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Матриця кореляцій")
plt.show()

correlation_with_target = correlation_matrix['Precipitation'].drop('Precipitation').abs()
top_features = correlation_with_target.nlargest(2).index.tolist()
X = data[top_features]
y = data['Precipitation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
coefficients = model.coef_
intercept = model.intercept_

print("Рівняння регресії:")
print(f"y = {intercept:.4f} + {' + '.join([f'{coef:.4f}*{name}' for coef, name in zip(coefficients, X.columns)])}")
y_pred = model.predict(X_test)
rss = np.sum((y_test - y_pred) ** 2)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r_squared = r2_score(y_test, y_pred)
n = len(y_test)
p = X_train.shape[1]
rse = np.sqrt(rss / (n - p - 1))

print("\nОцінка моделі:")
print(f"RSS: {rss:.2f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r_squared:.4f}")
print(f"RSE: {rse:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel("Реальні значення")
plt.ylabel("Прогнозовані значення")
plt.title("Реальні значення проти прогнозованих")
plt.show()
