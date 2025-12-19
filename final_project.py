import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# 1. 下載並讀取資料 (直接從 UCI 讀取)
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv"
df = pd.read_csv(url)

# --- 資料前處理 ---
# 選擇部分特徵用於演示
features = ['sex', 'age', 'Medu', 'Fedu', 'studytime', 'failures', 'schoolsup', 'famsup', 'goout', 'Dalc', 'Walc', 'absences', 'G1', 'G2']
target = 'G3'

data = df[features + [target]].copy()

# 類別變數編碼
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['schoolsup'] = le.fit_transform(data['schoolsup'])
data['famsup'] = le.fit_transform(data['famsup'])

# 定義 Target: G3 >= 10 為及格 (1), 否則為 0
data['pass'] = np.where(data['G3'] >= 10, 1, 0)

# --- 5. 監督式學習 (隨機森林) ---
X = data[features]
y = data['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# 畫圖：混淆矩陣
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Supervised Learning: Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 畫圖：特徵重要性
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 5))
feat_importances.nlargest(5).plot(kind='barh')
plt.title('Feature Importance (Top 5)')
plt.show()

# --- 6. 非監督式學習 (K-Means) ---
# 只選用行為特徵
behavior_features = ['studytime', 'goout', 'Dalc', 'Walc', 'absences']
X_cluster = data[behavior_features]

# 標準化
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# 尋找最佳 K 值 (Elbow Method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_cluster_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# 執行 K-Means (假設 K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_cluster_scaled)

# 畫圖：分群結果可視化 (取其中兩個特徵展示)
plt.figure(figsize=(6, 5))
plt.scatter(X_cluster_scaled[clusters == 0, 0], X_cluster_scaled[clusters == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X_cluster_scaled[clusters == 1, 0], X_cluster_scaled[clusters == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(X_cluster_scaled[clusters == 2, 0], X_cluster_scaled[clusters == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.title('Clusters of Students (StudyTime vs GoOut)')
plt.xlabel('Study Time (Scaled)')
plt.ylabel('Go Out (Scaled)')
plt.legend()

plt.show()
