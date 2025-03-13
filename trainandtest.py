from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

csv_file1 = 'OrderedIntervalTime.csv'
csv_file2 = 'OrderedPacketSize.csv'
csv_file3 = 'OrderedTCPWindow.csv'
csv_file4 = 'OrderedDirection.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)
df4 = pd.read_csv(csv_file4)

# 提取特征和标签集
L = df1['src_ip']
F1 = df1.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
F2 = df2.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
F3 = df3.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
F4 = df4.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
F = pd.concat([F1, F2, F3, F4], axis=1)

# 划分测试集
F_train, F_test, L_train, L_test = train_test_split(F, L, test_size=0.2, random_state=42)

# 训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # 使用100棵树
clf.fit(F_train, L_train)

# 在测试集上进行测试
L_pred = clf.predict(F_test)

# 计算准确率
accuracy = accuracy_score(L_test, L_pred)
print(f"Accuracy: {accuracy}")

# 进行单个样本预测（例如第一个测试样本）
sample = F_test.iloc[[0]]
predicted_label = clf.predict(sample)
print(f"预测的类别: {predicted_label}")