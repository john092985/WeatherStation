import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取已有的未归一化数据
df = pd.read_csv("ps.csv")

# 特征列（不含时间戳和标签）
feature_cols = [col for col in df.columns if col not in ['start_time', 'rainfall_12h_future']]

# 初始化 MinMaxScaler 并拟合 + 变换
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 保存归一化后的数据
df.to_csv("ps_normalized.csv", index=False)
print("✅ 已保存归一化数据集：ps_normalized.csv")