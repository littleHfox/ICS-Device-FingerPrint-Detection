import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

# 1. 读取 CSV 数据
csv_file1 = 'dft_output.csv'
csv_file2 = 'NormalizedPacketSize.csv'
csv_file3 = 'NormalizedTCPWindow.csv'
csv_file4 = 'OrderedDirection.csv'

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)
df3 = pd.read_csv(csv_file3)
df4 = pd.read_csv(csv_file4)

# 2. 预处理数据
# 提取 src_ip 作为标签
L = df1['src_ip'].values  

# 提取特征（去掉 IP 和端口等无关列）
drop_columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']
F1 = df1.drop(drop_columns, axis=1)
F2 = df2.drop(drop_columns, axis=1)
F3 = df3.drop(drop_columns, axis=1)
F4 = df4.drop(drop_columns, axis=1)

# 合并所有特征
F = pd.concat([F1, F2, F3, F4], axis=1)

# 3. 处理标签（转换为整数类别）
label_encoder = LabelEncoder()
L = label_encoder.fit_transform(L)  # 确保 L 是数值型

# 4. 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(F.values, L, test_size=0.2, random_state=42)

# 5. 创建 PyTorch 数据集
class CSVDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = CSVDataset(X_train, y_train)
test_dataset = CSVDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 6. 定义神经网络
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 添加 Dropout 0.3

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)  # Dropout 防止过拟合
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)  # Dropout 防止过拟合
        x = self.fc3(x)
        return x

# 获取输入特征维度和类别数
input_size = F.shape[1]  
num_classes = len(np.unique(L))  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, num_classes).to(device)

# 7. 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

epochs = 30
best_loss = float('inf')
patience = 5  # 允许连续 3 轮 Loss 变差
trigger_times = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    #scheduler.step()

    # Early stopping 机制
    if avg_loss < best_loss:
        best_loss = avg_loss
        trigger_times = 0  # 重新计数
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered")
            break  # 停止训练
    

# 8. 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# 9. 保存和加载模型
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")
