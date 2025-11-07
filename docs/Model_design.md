# User Story

现在拥有一个结构化的[区域数据集](./Data_format.md)，并且有一个明确的**预测目标：利用上游和下游区域的历史数据，预测下游区域未来的天气（12H累积降雨量）**。

基于提供的文件规范和 xarray 的使用提议，这里是一个参考方案，分为三个阶段：
- 数据ETL（提取、转换、加载）：将所有零散的 .txt 文件转换为一个（或多个）干净、对齐、可用于分析的 xarray.Dataset 文件。
- 模型设计（区域GNN）：设计一个受 [GraphCast启发的图神经网络（GNN）模型](./graph_cast.md)，该模型专门用于您定义的“上游-下游”区域预测任务。
- 训练与推理工作流：如何使用这个已处理的数据集来训练和运行您的模型。

**具体要求**：
- 阶段一：数据ETL（提取、转换、加载）目标：将 pre_Data/ 和 souding_data/ 目录下的所有文件，以及 Lon/Lat 文件，合并成一个单一的、以时间为索引的 processed_data.nc (NetCDF) 文件。工具：Python, xarray, pandas, numpy, glob
  - 1.1. 加载静态网格数据（Coordinates）首先，我们加载经纬度信息。您提供的 xarray 示例需要1D的坐标（lats, lons），而您的文件是2D的。加载文件：lat_matrix = np.loadtxt("Lat", delimiter=',')lon_matrix = np.loadtxt("Lon", delimiter=',')提取1D坐标：根据 Lat 文件的示例（每行值都相同），纬度坐标是恒定的。lats_coord = lat_matrix[:, 0]  （即取第一列）根据 Lon 文件的示例（每列值都相同），经度坐标是恒定的。lons_coord = lon_matrix[0, :]  （即取第一行）获取维度：num_lats = len(lats_coord)num_lons = len(lons_coord)
  - 1.2. 核心挑战：时间轴对齐您的 hpa 数据是 UTC (00:00, 12:00) 的瞬时值。您的 pre_Data 降雨数据是 BJT (UTC+8) (8-20, 20-8) 的累计值。我们必须将所有数据统一到一个时间轴上。UTC 是标准，我们将使用它。HPA 时间戳 (UTC)： ..._00 和 ..._12。这是我们的主时间轴。降雨量时间转换：yyyymmdd_8-20.txt (BJT 08:00 - 20:00)转换到 UTC： (yyyymmdd 08:00 BJT) -> (yyyymmdd 00:00 UTC)转换到 UTC： (yyyymmdd 20:00 BJT) -> (yyyymmdd 12:00 UTC)结论：yyyymmdd_8-20.txt 是在 yyyymmdd 12:00 UTC 这个时间点完成的12小时累计降雨。yyyymmdd_20-8.txt (前一日BJT 20:00 - 当日BJT 08:00)转换到 UTC： (前一日 20:00 BJT) -> (前一日 12:00 UTC)转换到 UTC： (当日 08:00 BJT) -> (当日 00:00 UTC)结论：yyyymmdd_20-8.txt 是在 yyyymmdd 00:00 UTC 这个时间点完成的12小时累计降雨。完美对齐：hpa 数据在 t 时刻的瞬时状态，对应了 pre_Data 在 t 时刻完成的过去12小时的累计降雨。
  - 1.3. 构建 xarray.Dataset我们将遍历所有HPA文件名来构建主时间轴，然后去匹配对应的降雨文件。
```Python
import xarray as xr
import pandas as pd
import numpy as np
import glob
import os

# --- 1. 加载坐标 ---
lats_coord = np.loadtxt("Lat", delimiter=',')[:, 0]
lons_coord = np.loadtxt("Lon", delimiter=',')[:, 0]
num_lats = len(lats_coord)
num_lons = len(lons_coord)

# --- 2. 扫描文件并构建时间轴 ---
hpa_files = glob.glob("souding_data/*.txt")
timestamps_utc = set()
for f in hpa_files:
    filename = os.path.basename(f)
    parts = filename.split('_')
    # 提取 yyyymmdd_HH
    timestamp_key = f"{parts[0]}_{parts[1]}"
    timestamps_utc.add(timestamp_key)

# 排序并转换为Pandas DatetimeIndex (我们的 "time" 坐标)
sorted_keys = sorted(list(timestamps_utc))
times_coord = pd.to_datetime(sorted_keys, format="%Y%m%d_%H")

# --- 3. 定义变量和高度 (我们的 "level" 坐标) ---
levels_coord = [100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000]
hpa_types = ["DPT", "GPH", "TEM", "U", "V"]

# --- 4. 初始化空的numpy数组 ---
# (time, level, lat, lon)
hpa_data_store = {
    var: np.full((len(times_coord), len(levels_coord), num_lats, num_lons), np.nan, dtype=np.float32)
    for var in hpa_types
}
# (time, lat, lon)
precip_data = np.full((len(times_coord), num_lats, num_lons), np.nan, dtype=np.float32)

# --- 5. 循环加载数据 ---
for t_idx, timestamp_key in enumerate(sorted_keys):
    date_str, time_str = timestamp_key.split('_') # e.g., "20230101", "12"
    
    # 5.1 加载 HPA 数据
    for l_idx, level in enumerate(levels_coord):
        for var in hpa_types:
            hpa_filename = f"souding_data/{date_str}_{time_str}_hpa{level}_{var}.txt"
            if os.path.exists(hpa_filename):
                data = np.loadtxt(hpa_filename, delimiter=',')
                hpa_data_store[var][t_idx, l_idx, :, :] = data

    # 5.2 加载匹配的降雨数据
    precip_filename = ""
    if time_str == "12": # 对应 8-20 BJT
        precip_filename = f"pre_Data/{date_str}_8-20.txt"
    elif time_str == "00": # 对应 20-8 BJT
        precip_filename = f"pre_Data/{date_str}_20-8.txt"
        
    if os.path.exists(precip_filename):
        data = np.loadtxt(precip_filename, delimiter=',')
        data[data < 0] = 0.0 # 清理负值
        precip_data[t_idx, :, :] = data

# --- 6. 创建 Xarray DataArrays ---
data_arrays = {}

# HPA 变量
for var in hpa_types:
    data_arrays[var] = xr.DataArray(
        hpa_data_store[var],
        dims=["time", "level", "lat", "lon"],
        coords={"time": times_coord, "level": levels_coord, "lat": lats_coord, "lon": lons_coord},
        name=var
    )

# 降雨变量
data_arrays["precipitation"] = xr.DataArray(
    precip_data,
    dims=["time", "lat", "lon"],
    coords={"time": times_coord, "lat": lats_coord, "lon": lons_coord},
    name="precipitation"
)

# --- 7. 合并并保存 ---
ds = xr.Dataset(data_arrays)

# 添加元数据
ds.attrs["description"] = "Regional atmospheric and precipitation data."
ds.attrs["region"] = "East [70, 140], North [10, 60]"
ds.attrs["resolution"] = "0.25 degrees"
ds.attrs["hpa_time_info"] = "Instantaneous UTC"
ds.attrs["precipitation_time_info"] = "12-hour accumulated ending at timestamp (BJT 8-20 or 20-8)"

# 保存到高效的 NetCDF 文件
ds.to_netcdf("processed_regional_data.nc")
print("数据处理完成，已保存至 processed_regional_data.nc")
```

- 阶段二：模型设计（区域 GNN）现在您有了一个干净的 processed_regional_data.nc 文件，可以开始设计模型了。我们将采用您之前讨论的“上游-下游”GNN架构。目标：预测下游区域在 $t+1$ 时刻的降雨量。输入：上游和下游区域在 $t$ 和 $t-1$ 时刻的所有HPA变量（TEM, DPT, GPH, U, V）和降雨量。   
  - 2.1. 定义图（Nodes and Edges）定义区域：这是您的第一个关键决策。您需要定义两个区域的经纬度范围。downstream_bounds = {"lat_min": 25, "lat_max": 40, "lon_min": 110, "lon_max": 125} (例如，华东地区)upstream_bounds = {"lat_min": 25, "lat_max": 50, "lon_min": 70, "lon_max": 110} (例如，中亚到华中)提取节点：加载 ds = xr.open_dataset("processed_regional_data.nc")downstream_nodes = ds.where((ds.lat >= 25) & ... & (ds.lon <= 125), drop=True)upstream_nodes = ds.where((ds.lat >= 25) & ... & (ds.lon <= 110), drop=True)构建图 (Graph)：节点 (Nodes)：$V = V_{downstream} \cup V_{upstream}$。每个节点都是一个 0.25° 的网格点。边 (Edges)：内部边 (Intra-domain)：连接下游区域 $V_{downstream}$ 内部的邻近节点（例如，8个最近邻）。这用于模拟下游区域的内部物理过程。边界边 (Inter-domain)：连接上游区域 $V_{upstream}$ 和下游区域 $V_{downstream}$ 的边。这是关键。一个简单的方法是：对于下游区域中靠近边界（例如，经度 < 111°）的每个节点，将其连接到上游区域中最近的 $k$ 个节点（例如 $k=32$）。这模拟了平流和边界输入。
  
  - 2.2. 模型架构 (Encoder-Processor-Decoder)这是一个混合了 GraphCast/GenCast 思想的区域模型。输入特征 ($X$)：在时刻 $t$ 和 $t-1$ (12小时间隔)，堆叠所有变量。变量：5个HPA变量 $\times$ 11个高度 + 1个降雨变量 = 56个通道。总输入： $56 \times 2 = 112$ 个特征通道。形状：(num_nodes, 112)目标标签 ($Y$)：在时刻 $t+1$ 的 precipitation 变量。形状：(num_downstream_nodes, 1)编码器 (Encoder)：一个简单的 MLP (多层感知机)，将每个节点的 (112) 维输入特征，映射到一个高维的隐藏状态 (e.g., 256 维)。X_latent = MLP(X)处理器 (Processor)：一个堆叠的图神经网络（例如10-16层 GNN）。它在您 2.1 节定义的混合图上运行。在每一层，下游节点从其内部邻居和上游邻居收集信息并更新自身状态。H = GNN_Layer_1(X_latent, Edges)H = GNN_Layer_2(H, Edges) + H (使用残差连接)...H_final = GNN_Layer_16(H, Edges) + H解码器 (Decoder)：一个 MLP，仅作用于 $V_{downstream}$ 节点的最终隐藏状态 H_final。它将 (256) 维的隐藏状态解码回一个物理预测值。Y_pred = Decoder_MLP(H_final[downstream_indices])输出形状：(num_downstream_nodes, 1)

- 阶段三：训练与推理工作流
  - 3.1. 数据集准备 (Pytorch / Tensorflow)加载数据：ds = xr.open_dataset("processed_regional_data.nc")标准化：计算训练集（例如2010-2018年）上所有112个输入特征的均值和标准差，并将其应用于所有数据。这是至关重要的。时间窗口：使用滑动窗口创建样本。X = (数据在 $t-12h$, 数据在 $t$)Y = (降雨在 $t+12h$)拆分：按时间拆分训练/验证/测试集。绝不能随机打乱，必须保持时序。train_ds = ds.sel(time=slice('2010-01-01', '2018-12-31'))val_ds = ds.sel(time=slice('2019-01-01', '2019-12-31'))test_ds = ds.sel(time=slice('2020-01-01', '2020-12-31'))
  - 3.2. 训练损失函数：不要用简单的 MSE (L2 Loss)：降雨数据是高度稀疏（很多0）且长尾的（少数极端值）。MSE 会导致模型只预测“平均毛毛雨”，无法捕捉暴雨。推荐方案A (简单)：Weighted MSE。计算损失时，给大降雨量（例如 > 10mm）的网格点更高的权重。推荐方案B (GenCast/概率思想)：CRPS Loss (Continuous Ranked Probability Score) 1。这要求您的解码器输出一个分布（例如均值和方差），而不是一个单一值。这能更好地处理不确定性。推荐方案C (实用)：ZILN (Zero-Inflated Log-Normal) 损失。这是一种两部分模型：BCE Loss (二元交叉熵)：预测降雨概率 $P(\text{rain} > 0)$。Log-Normal Loss：预测 $\log(\text{rain_amount} | \text{rain} > 0)$。优化器：使用 AdamW。
  - 3.3. 推理 (Inference)当您需要为 $t+1$ 时刻进行预测时：加载 processed_regional_data.nc。提取 $t$ 和 $t-1$ 时刻的112个特征通道。应用训练时保存的标准化参数。将数据输入到您训练好的GNN模型中。解码器将输出 $t+1$ 时刻的12小时累计降雨量预测图。
- 阶段四：结果评估与效果展示
  - 4.1 在测试集数据上（比如2020.6.1 以后的数据）上进行结果评估，给出平均的MSE；
  - 4.2 撰写可视化效果，在下游的区域上，给定一个时间点，使用前面两个时间步进行预测，然后与真实结果进行可视化对比。