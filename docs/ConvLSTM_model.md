# 1. 背景说明

这是一个非常有实际意义的气象应用场景，特别是用于论证“上游天气系统对下游的影响”。针对已有的需求（Dense预测、时空相关性、显存受限），设计了一套基于深度学习的解决方案。核心思路是采用 ConvLSTM (卷积长短期记忆网络) 结合 Encoder-Decoder (U-Net) 架构。ConvLSTM：利用卷积核提取空间特征（考虑相邻网格），利用LSTM处理时间序列（考虑过去3天的数据）。U-Net结构：保证输出的分辨率与输入一致（Dense Prediction），适合网格化预测。显存优化：通过通道堆叠（Channel Stacking）处理多层探空数据，避免使用昂贵的3D卷积。

## 1.1 detail 

### 一、 数据预处理与输入构建在12G显存下，数据的组织方式决定了是否能跑起来。

1. 特征工程 (Channel Construction)气象数据是3D（多层）的，但在深度学习中，为了节省显存，我们通常将“高度层”展平为“通道（Channel）”。假设每个网格点有 $L$ 层(本方案中11个)探空数据，每层有 $V$ 个（5个）变量（如露点温度、温度、U/V风、位势高度），加上地面 $1$ 个变量（累计降雨）。通道总数 ($C_{in}$) = $L \times V + 1$。例如：11层大气 × 5个变量 + 1个地面降雨 = 56个通道。这对于12G显存完全没压力。
2. 数据形状 (Tensor Shape)输入张量的形状应为：$[B, T, C_{in}, H, W]$B (Batch Size): 批次大小（建议设为 4 或 8）。T (Time Steps): 时间步长。过去3天，12小时一次，则 $T=6$。H, W (Grid Size): 区域的网格大小（例如 根据实际区域大小来设计）。

### 二、 网络模型设计：ST-RainNet (Spatio-Temporal Rain Net)我们将构建一个轻量级的 ConvLSTM U-Net。

模型架构图示
```Code snippetgraph LR
    Input[过去6帧序列] --> Encoder
    subgraph Encoder
    CL1[ConvLSTM 1] --> Pool1[Downsample]
    Pool1 --> CL2[ConvLSTM 2]
    end
    
    CL2 --> Bottleneck[ConvLSTM Bottleneck]
    
    subgraph Decoder
    Bottleneck --> UP1[Upsample]
    UP1 --> CAT[Concat with CL1 Output]
    CAT --> CL3[ConvLSTM 3]
    end
    
    CL3 --> Head[Conv2D 1x1]
    Head --> Output[未来1帧降雨图]
```

**PyTorch 代码实现框架**

```Python
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """ 简化的 ConvLSTM 单元，处理时空数据 """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # 拼接输入和上一时刻状态
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ST_RainNet(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(ST_RainNet, self).__init__()
        
        # 1. Encoder (捕捉特征并下采样)
        self.encoder_1 = ConvLSTMCell(input_channels, hidden_channels[0], 3, True)
        self.pool = nn.MaxPool2d(2) # 增大感受野，节省显存
        
        # 2. Bottleneck (处理压缩后的时空特征)
        self.bottleneck = ConvLSTMCell(hidden_channels[0], hidden_channels[1], 3, True)
        
        # 3. Decoder (上采样恢复分辨率)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 输入是 bottleneck输出 + encoder_1的跳跃连接
        self.decoder_1 = ConvLSTMCell(hidden_channels[1] + hidden_channels[0], hidden_channels[0], 3, True)
        
        # 4. Output Head (输出预测)
        # 将最后时刻的隐藏状态映射为降雨量 (1 channel)
        self.final_conv = nn.Conv2d(hidden_channels[0], 1, kernel_size=1)
        self.relu = nn.ReLU() # 降雨量必须 >= 0

    def forward(self, x):
        """
        x shape: [Batch, Time, Channels, Height, Width]
        """
        b, t, c, h, w = x.size()
        
        # 初始化状态 (h, c) 为全0
        h1, c1 = self._init_state(b, self.encoder_1.hidden_dim, h, w)
        h2, c2 = self._init_state(b, self.bottleneck.hidden_dim, h//2, w//2)
        
        # 循环处理时间步
        # 注意：这里我们只需要最后时刻的输出用于预测，或者可以用所有时刻做序列预测
        # 为了节省显存，我们这里简化为只用最后时刻的状态做预测
        
        # 保存 Skip Connection 的列表
        skip_connections = [] 
        
        for step in range(t):
            current_input = x[:, step, :, :, :]
            
            # Encoder Layer 1
            h1, c1 = self.encoder_1(current_input, (h1, c1))
            
            # Downsample
            encoded_feat = self.pool(h1)
            
            # Bottleneck
            h2, c2 = self.bottleneck(encoded_feat, (h2, c2))
            
        # 解码阶段 (只使用最后的时间步信息进行预测 T+1)
        # Upsample bottleneck output
        upsampled_h2 = self.upsample(h2)
        
        # Skip Connection: Concatenate
        concat_feat = torch.cat([upsampled_h2, h1], dim=1)
        
        # Decoder Layer (这里简化了，Decoder通常不需要时间循环，除非你要输出序列)
        # 我们把解码器当作普通的CNN处理当前特征，或者再过一层ConvLSTMCell
        h_out, _ = self.decoder_1(concat_feat, (h1, c1)) # 复用h1大小的状态
        
        # Final Prediction
        prediction = self.final_conv(h_out)
        return self.relu(prediction) # 保证非负

    def _init_state(self, b, hidden, h, w):
        return (torch.zeros(b, hidden, h, w).cuda(), 
                torch.zeros(b, hidden, h, w).cuda())
```
三、 针对两个实验的训练策略1为了论证“上游对下游的影响”，你需要精心设计输入数据的空间结构。

#### 实验 1：

仅区域 1 (Baseline) 输入维度: $H \times W$ (区域1的大小)。
逻辑: 模型只能看到区域1内部的历史演变。
目的: 建立基准线。
#### 实验 2：
区域 1 + 区域 2 (Comparison)数据拼接 
(关键点):如果区域2在区域1的西边（上游），你应该将数据在空间上进行拼接（Spatial Concatenation）。新网格宽度 $W_{new} = W_{area1} + W_{area2}$。输入维度: $H \times W_{new}$。

Masking (掩码机制):虽然输入是两个区域，但你的预测目标（Ground Truth）仍然只针对区域1。在计算 Loss 时，只计算区域1对应的网格部分的Loss，忽略区域2的预测结果。
显存控制: 如果拼接后网格太大，可以使用 Stride (步长) 卷积先进行降维，或者在输入前对数据进行重采样（Resample）降低分辨率。

### 四、 训练细节与显存优化 (12G GPU)

1. **混合精度训练 (AMP)**:
务必使用 torch.cuda.amp。这能将显存占用减少近一半，并加速训练。
scaler = torch.cuda.amp.GradScaler()

2. **Loss 函数的选择**:

降雨数据通常有很多0值（稀疏），且极端值很大（长尾分布）。
单纯的MSE会导致模型倾向于预测0或者模糊的平均值。
推荐 Loss: Log-Cosh Loss 或者 Weighted MSE。公式:
$$L = \frac{1}{N} \sum (y_{pred} - y_{true})^2 \times (1 + \alpha \times y_{true})$$
其中 $\alpha$ 是权重，给大雨更大的惩罚权重，迫使模型关注极端天气。

3. **数据归一化 (Normalization)**:
这一点至关重要。温度/气压/风速: 使用 Z-Score 标准化 ($\frac{x - \mu}{\sigma}$)。降雨量: 使用 Log1p 变换 ($\log(x+1)$) 压缩数值范围，因为降雨量级跨度大。预测后再用 $e^x - 1$ 还原。

4. **超参数建议**:
5. Batch Size: 4 或 8 (如果爆显存，设为1，然后使用 gradient_accumulation_steps=4 来模拟Batch=4)。Hidden Channels: 开始尝试 [32, 64]，不要设太大。Learning Rate: 1e-3 或 1e-4，配合 AdamW 优化器。

### 五、 论证方法 (Evaluation)

为了有力地证明上游数据的影响，你需要对比以下指标：RMSE (均方根误差): 区域1内的整体误差。CSI (Critical Success Index): 针对不同降雨等级（小雨、中雨、大雨）的命中率。通常上游数据对大雨或突发性天气的预测提升最明显。可视化:画出区域1在T+1时刻的 真实降雨图 vs 实验1预测图 vs 实验2预测图。做一张 Difference Map（预测图 - 真实图），展示实验2的误差热力图是否比实验1颜色更淡。