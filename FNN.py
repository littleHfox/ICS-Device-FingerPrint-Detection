import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

target_ip = '192.168.10.150'

# csv_file1 = '2_formatted/dft_IntervalTime.csv'
csv_file1 = '2_formatted/dft_IntervalTime_mod.csv'
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

# 假设你已经准备好了特征 X 和标签 y
X_np = X.values.astype(np.float32)
y_np = y.values

# 标签编码为整数
le = LabelEncoder()
y_encoded = le.fit_transform(y_np)

# ===== 在交叉验证之前先进行随机过采样 =====
# 过采样
sampling_strategy = {}
unique_labels, counts = np.unique(y_encoded, return_counts=True)
for label, count in zip(unique_labels, counts):
    if count < 5:
        sampling_strategy[label] = 5
    else:
        sampling_strategy[label] = count
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)

# ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_np, y_encoded)

# 转为 Tensor
X_tensor = torch.tensor(X_resampled, dtype=torch.float32)
y_tensor = torch.tensor(y_resampled, dtype=torch.long)

# 设置交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 定义模型
class FNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# 超参数
num_classes = len(np.unique(y_encoded))
input_dim = X_np.shape[1]
epochs = 30
batch_size = 32
learning_rate = 0.001

# 存储每一折的指标
acc_list, f1_list, precision_list, recall_list = [], [], [], []

# 交叉验证循环
for fold, (train_idx, test_idx) in enumerate(skf.split(X_resampled, y_resampled)):
    print(f"\nFold {fold+1}:")

    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

    model = FNN(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === 模型训练 ===
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]

            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    # === 模型评估 ===
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test)
        y_pred = torch.argmax(y_pred_logits, dim=1).numpy()
        y_true = y_test.numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"准确率 Accuracy: {acc:.4f}")
    print(f"精准率 Precision (macro): {precision:.4f}")
    print(f"召回率 Recall (macro): {recall:.4f}")
    print(f"F1 分数 (macro): {f1:.4f}")

# 最终结果
print("\n==== 5折交叉验证结果 ====")
print(f"平均准确率: {np.mean(acc_list):.4f}")
print(f"平均精准度 (macro): {np.mean(precision_list):.4f}")
print(f"平均召回率 (macro): {np.mean(recall_list):.4f}")
print(f"平均 F1 分数 (macro): {np.mean(f1_list):.4f}")