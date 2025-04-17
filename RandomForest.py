from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

target_ip = '192.168.10.150'

csv_file1 = '2_formatted/dft_IntervalTime.csv'
# csv_file2 = '2_formatted/dft_PacketSize.csv'
# csv_file3 = '2_formatted/dft_TCPWindow.csv'
# csv_file4 = '2_formatted/dft_Direction.csv'

# csv_file1 = '2_formatted/Origin_IntervalTime.csv'
csv_file2 = '2_formatted/NormalizedPacketSize.csv'
csv_file3 = '2_formatted/NormalizedTCPWindow.csv'
csv_file4 = '2_formatted/OrderedDirection.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)
df4 = pd.read_csv(csv_file4)

# 提取特征和标签集
y = df1['src_ip']
# y = (df1['src_ip'] == target_ip).astype(int)
# X1 = df1.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port'], axis=1)
X1 = df1.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X2 = df2.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X3 = df3.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X4 = df4.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X = pd.concat([X1, X2, X3, X4], axis=1)

# 过采样
# sampling_strategy = {}
# for label, count in y.value_counts().items():
#     if count < 5:
#         sampling_strategy[label] = 5
#     else:
#         sampling_strategy[label] = count
# ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 在采样后的数据上做交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化模型和记录列表
model = RandomForestClassifier(random_state=42)
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for train_index, test_index in skf.split(X_resampled, y_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_list.append(accuracy_score(y_test, y_pred))
    precision_list.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
    recall_list.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
    f1_list.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

# 输出最终结果
print("五折交叉验证结果：")
print(f"平均准确率：{np.mean(accuracy_list):.4f}")
print(f"平均精准率（macro）：{np.mean(precision_list):.4f}")
print(f"平均召回率（macro）：{np.mean(recall_list):.4f}")
print(f"平均F1分数（macro）：{np.mean(f1_list):.4f}")

