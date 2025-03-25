from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

target_ip = '192.168.10.150'

csv_file1 = '1/NormalizedIntervalTime.csv'
csv_file2 = '1/NormalizedPacketSize.csv'
csv_file3 = '1/NormalizedTCPWindow.csv'
csv_file4 = '1/OrderedDirection.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)
df4 = pd.read_csv(csv_file4)

# 提取特征和标签集
L = df1['src_ip']
#L = (df1['src_ip'] == target_ip).astype(int)
F1 = df1.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
F2 = df2.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
F3 = df3.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
F4 = df4.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
F = pd.concat([F1, F2, F3, F4], axis=1)



# 划分测试集
#F_train, F_test, L_train, L_test = train_test_split(F, L, test_size=0.3, random_state=42)

# 训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)  # 使用100棵树
#clf.fit(F_train, L_train)

# 使用五重交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 进行交叉验证，评估准确率
scores = cross_val_score(clf, F, L, cv=cv, scoring='accuracy')

# 在测试集上进行测试
# L_pred = clf.predict(F_test)

# 计算准确率
# accuracy = accuracy_score(L_test, L_pred)
# print(f"Accuracy: {accuracy}")

# 进行单个样本预测（例如第一个测试样本）
# sample = F_test.iloc[[0]]
# predicted_label = clf.predict(sample)
# print(f"预测的类别: {predicted_label}")

# 输出每次交叉验证的准确率
print(f'五重交叉验证的准确率：{scores}')

# 输出交叉验证的平均准确率
print(f'平均准确率：{scores.mean()}')