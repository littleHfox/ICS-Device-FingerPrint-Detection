import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

# === 读取数据 ===
csv_file1 = '2_formatted/dft_IntervalTime_mod.csv'
csv_file2 = '2_formatted/NormalizedPacketSize.csv'
csv_file3 = '2_formatted/NormalizedTCPWindow.csv'
csv_file4 = '2_formatted/OrderedDirection.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)
df4 = pd.read_csv(csv_file4)

y = df1['src_ip']
X1 = df1.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X2 = df2.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X3 = df3.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)
X4 = df4.drop(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], axis=1)

X = pd.concat([X1, X2, X3, X4], axis=1)
X_np = X.values.astype(np.float32)
le = LabelEncoder()
y_encoded = le.fit_transform(y.values)

# === 折前过采样 ===
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_np, y_encoded)

# === 转换为 CNN 输入形状 (N, C, L) ===
X_tensor = torch.tensor(X_resampled, dtype=torch.float32)
y_tensor = torch.tensor(y_resampled, dtype=torch.long)
X_tensor = X_tensor.unsqueeze(1)  # 添加通道维度 (N, 1, L)

# === 模型定义 ===
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(identity)
        return self.relu(out + identity)

class ResNet1D(nn.Module):
    def __init__(self, input_length, num_classes):
        super().__init__()
        self.layer1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.res1 = ResBlock1D(32, 32)
        self.res2 = ResBlock1D(32, 64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# === 超参数设置 ===
epochs = 30
batch_size = 32
learning_rate = 0.001
input_length = X_tensor.shape[2]
num_classes = len(np.unique(y_resampled))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === K折交叉验证 ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, precision_list, recall_list, f1_list = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_tensor, y_tensor)):
    print(f"\nFold {fold+1}")
    X_train, X_test = X_tensor[train_idx].to(device), X_tensor[test_idx].to(device)
    y_train, y_test = y_tensor[train_idx].to(device), y_tensor[test_idx].to(device)

    model = ResNet1D(input_length, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # === 训练 ===
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # === 评估 ===
    model.eval()
    with torch.no_grad():
        y_pred_logits = model(X_test)
        y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"准确率: {acc:.4f}, 精准率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")

# === 最终输出 ===
print("\n==== 5折交叉验证结果 ====")
print(f"平均准确率: {np.mean(acc_list):.4f}")
print(f"平均精准率: {np.mean(precision_list):.4f}")
print(f"平均召回率: {np.mean(recall_list):.4f}")
print(f"平均F1分数: {np.mean(f1_list):.4f}")
