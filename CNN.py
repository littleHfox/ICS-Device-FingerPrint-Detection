import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

# ===== 设备设置 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== 数据加载 =====
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

X = X.values.astype(np.float32)
y = y.values

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y)

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
X_resampled, y_resampled = ros.fit_resample(X, y_encoded)

# ===== 转Tensor + 调整形状 =====
X_resampled = torch.tensor(X_resampled, dtype=torch.float32).to(device)
y_resampled = torch.tensor(y_resampled, dtype=torch.long).to(device)

# CNN需要给输入加 channel 维度 (N, C, L)
X_resampled = X_resampled.unsqueeze(1)  # 在第1维插入通道数，原本(N, feature) → (N, 1, feature)

# ===== CNN定义 =====
class ImprovedCNN(nn.Module):
    def __init__(self, input_len, num_classes):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # 将特征图池化为 (batch, channels, 1)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ===== 超参数 =====
input_dim = X.shape[1]
num_classes = len(np.unique(y_encoded))
epochs = 30
batch_size = 32
learning_rate = 0.001

# ===== 五折交叉验证 =====
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_list, f1_list, precision_list, recall_list = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_resampled.cpu(), y_resampled.cpu())):
    print(f"\n=== Fold {fold+1} ===")
    
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

    model = ImprovedCNN(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === 训练 ===
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    # === 测试 ===
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    # === 计算指标 ===
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"准确率: {acc:.4f} 精准率: {precision:.4f} 召回率: {recall:.4f} F1: {f1:.4f}")

# ===== 汇总 =====
print("\n==== 5折交叉验证结果 ====")
print(f"平均准确率: {np.mean(acc_list):.4f}")
print(f"平均精准率 (macro): {np.mean(precision_list):.4f}")
print(f"平均召回率 (macro): {np.mean(recall_list):.4f}")
print(f"平均 F1 分数 (macro): {np.mean(f1_list):.4f}")
