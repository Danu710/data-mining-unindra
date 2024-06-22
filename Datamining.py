import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Memuat dataset penjualan
data = pd.read_csv('walmart.csv')

# Eksplorasi Data
print(data.head())
print(data.describe())
print(data.info())

# Visualisasi Data
sns.histplot(data['Weekly_Sales'])
plt.title('Distribusi Penjualan Mingguan')
plt.show()

# Mengatasi data yang hilang
data = data.dropna()

# Encoding data kategorikal
label_encoder = LabelEncoder()
data['Store'] = label_encoder.fit_transform(data['Store'])
data['Dept'] = label_encoder.fit_transform(data['Dept'])
data['Type'] = label_encoder.fit_transform(data['Type'])

# Split data menjadi training dan testing set
X = data.drop(columns=['Weekly_Sales'])
y = data['Weekly_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model regresi linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualisasi Hasil Prediksi vs Aktual
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
