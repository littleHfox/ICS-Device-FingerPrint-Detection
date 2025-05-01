import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd

# 数据加载
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

# 模型定义（ResNet 风格 FNN）
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)

class ResNetFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.res1 = ResBlock(128)
        self.res2 = ResBlock(128)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.dropout(x)
        return self.output(x)

# 超参数设置
epochs = 30
batch_size = 32
learning_rate = 0.001
input_dim = X_np.shape[1]
num_classes = len(np.unique(y_encoded))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 交叉验证和每折过采样
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_list, precision_list, recall_list, f1_list = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_np, y_encoded)):
    print(f"\nFold {fold+1}")
    X_train, y_train = X_np[train_idx], y_encoded[train_idx]
    X_test, y_test = X_np[test_idx], y_encoded[test_idx]

    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

    X_train_tensor = torch.tensor(X_train_res, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_res, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    model = ResNetFNN(input_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 训练
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(X_train_tensor), batch_size):
            xb = X_train_tensor[i:i+batch_size]
            yb = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # 评估
    model.eval()
    with torch.no_grad():
        y_logits = model(X_test_tensor)
        y_pred = torch.argmax(y_logits, dim=1).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"准确率: {acc:.4f}, 精准率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")

# 总结
print("\n==== 5折交叉验证====")
print(f"平均准确率: {np.mean(acc_list):.4f}")
print(f"平均精准率: {np.mean(precision_list):.4f}")
print(f"平均召回率: {np.mean(recall_list):.4f}")
print(f"平均F1分数: {np.mean(f1_list):.4f}")
