import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

# ====== 数据准备 ======
# 你的CSV路径
csv_file1 = '2_formatted/dft_IntervalTime_mod.csv'
csv_file2 = '2_formatted/NormalizedPacketSize.csv'
csv_file3 = '2_formatted/NormalizedTCPWindow.csv'
csv_file4 = '2_formatted/OrderedDirection.csv'

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

X_np = X.values.astype(np.float32)
y_np = y.values

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y_np)

# ====== 硬件设备 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# ====== 模型定义（优化版 CNN）======
class ImprovedCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64 * input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加 channel 维度
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 展平成全连接
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ====== 超参数 ======
input_dim = X_np.shape[1]
num_classes = len(np.unique(y_encoded))
batch_size = 32
epochs = 30
learning_rate = 0.001
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ====== 五折交叉验证 ======
acc_list, precision_list, recall_list, f1_list = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_np, y_encoded)):
    print(f"\n=== Fold {fold+1} ===")

    # 划分数据
    X_train, X_test = X_np[train_idx], X_np[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # === 只对 train 做过采样 ===
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # 转为 tensor
    X_train_tensor = torch.tensor(X_train_resampled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # 模型
    model = ImprovedCNN(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === 训练 ===
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            xb = X_train_tensor[i:i+batch_size]
            yb = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    # === 测试 ===
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

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
    print(f"F1分数 F1 (macro): {f1:.4f}")

# ====== 总结输出 ======
print("\n=== 五折交叉验证整体结果 ===")
print(f"平均准确率: {np.mean(acc_list):.4f}")
print(f"平均精准率: {np.mean(precision_list):.4f}")
print(f"平均召回率: {np.mean(recall_list):.4f}")
print(f"平均F1分数: {np.mean(f1_list):.4f}")
