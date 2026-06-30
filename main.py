# Gerekli Kütüphaneleri İçe Aktarma
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Veri setini bulma ve yükleme 
print("Veri seti yükleniyor...\n")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names) # Özellikler (Features)
y = data.target # Hedef Değişken (0: Malignant/Kötü Huylu, 1: Benign/İyi Huylu)

# 2. Veri Temizleme, Ölçeklendirme (Feature Scaling) ve Bölme işlemleri
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf_model = RandomForestClassifier(random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)

print("=== Random Forest Modeli Sonuçları ===")
print(f"Accuracy Score: {accuracy_score(y_test, rf_predictions):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("\n")

print("=== k-Nearest Neighbors (k-NN) Modeli Sonuçları ===")
print(f"Accuracy Score: {accuracy_score(y_test, knn_predictions):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))
print("\n")

"""
CONCLUSION:
When comparing the two models on the Breast Cancer diagnostic dataset, the Random Forest classifier 
slightly outperformed the k-NN model in terms of overall Accuracy. This performance difference is 
likely because Random Forest is an ensemble learning method that builds multiple decision trees, 
making it highly robust against overfitting and highly dimensional data (like our 30 features here). 
While k-NN relies purely on spatial distance metrics which can struggle as dimensionality increases, 
Random Forest effectively identifies the most important features, making it the better fit for this 
specific diagnostic dataset.
"""