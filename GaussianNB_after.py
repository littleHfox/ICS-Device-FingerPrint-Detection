from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

target_ip = '192.168.10.150'

# csv文件路径
# csv_file1 = '2_formatted/Origin_IntervalTime.csv'
csv_file1 = '2_formatted/dft_IntervalTime_mod.csv'
csv_file2 = '2_formatted/NormalizedPacketSize.csv'
csv_file3 = '2_formatted/NormalizedTCPWindow.csv'
csv_file4 = '2_formatted/OrderedDirection.csv'

# 读数据
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)
df4 = pd.read_csv(csv_file4)

# 提取特征和标签
y = df1['src_ip']
X1 = df1.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X2 = df2.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X3 = df3.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X4 = df4.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X = pd.concat([X1, X2, X3, X4], axis=1)

# StratifiedKFold交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化模型和记录列表
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n=== Fold {fold+1} ===")

    # 按照索引划分
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # === 在训练集上单独过采样 ===
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # 训练和预测
    model = GaussianNB()
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    # 保存结果
    accuracy_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"准确率 Accuracy: {acc:.4f}")
    print(f"精准率 Precision: {precision:.4f}")
    print(f"召回率 Recall: {recall:.4f}")
    print(f"F1分数 F1: {f1:.4f}")

# 输出最终平均结果
print("\n=== 五折交叉验证整体结果 ===")
print(f"平均准确率：{np.mean(accuracy_list):.4f}")
print(f"平均精准率（macro）：{np.mean(precision_list):.4f}")
print(f"平均召回率（macro）：{np.mean(recall_list):.4f}")
print(f"平均F1分数（macro）：{np.mean(f1_list):.4f}")
