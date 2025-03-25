import pandas as pd
import numpy as np

# 读取 CSV 文件
file_path = "1/NormalizedIntervalTime.csv"  # 你的 CSV 文件路径
df = pd.read_csv(file_path)

w = 12
cut = 5
# 计算DFT并存储
def compute_dft(row):
    X = np.fft.fft(row)  # 计算DFT
    return pd.Series(X)  # 返回完整的DFT结果

df_out = df.iloc[:, 0:5]
for i in range(17-w):
    slid_window = df.iloc[:, i+5:i+5+w]

    # 确保数据是数值型
    slid_window = slid_window.apply(pd.to_numeric, errors='coerce')

    # 对每一行计算DFT
    dft_results = slid_window.apply(compute_dft, axis=1)

    # 拆分复数的实部和虚部，方便存储
    dft_real = dft_results.map(lambda x: x.real)  # 获取实部
    dft_imag = dft_results.map(lambda x: x.imag)  # 获取虚部

    # 将 DFT 结果合并回原数据框
    dft_real.columns = [f"DFT_Real_{j}" for j in range(4*i, 4*i+w)]
    dft_imag.columns = [f"DFT_Imag_{j}" for j in range(4*i, 4*i+w)]

    # 将DFT结果合并回原数据框（不要虚部）
    df_out = pd.concat([df_out, dft_real.iloc[:, 0:cut]], axis=1)

# 保存结果
output_path = "1/dft_output.csv"
df_out.to_csv(output_path, index=False)

print(f"DFT 计算完成，结果已保存到 {output_path}")
