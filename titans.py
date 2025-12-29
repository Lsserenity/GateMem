import torch
from torch import nn
from torch.func import functional_call
from torch.nn.functional import normalize
from memory_network import NeuralMemory
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# titan的MAC层实现
class MACTitan(nn.Module):

    # MAC连接attention输出，neural memory输出，persistent memory输出，dynamic cheatsheet memory输出
    def __init__(self, hidden_dim, seq_len, pm_len, dc_len, layers_num = 2, alpha = 0.999, eta = 0.8, theta = 0.3):
        super().__init__()

        # 初始化各个维度和参数
        self.seq_len = seq_len
        self.pm_len = pm_len
        self.dc_len = dc_len
        self.hidden_dim = hidden_dim
        # 计算中间维度
        self.inter_dim = (dc_len, pm_len + 2 * hidden_dim)

        # 持久化记忆层
        self.persistent_memory = nn.Parameter(torch.randn((pm_len, self.hidden_dim)))
        # 注意力层
        self.att_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            activation=nn.SiLU(),
            batch_first=True
        )
        # neural memory查询层
        self.nm_module = NeuralMemory(
            input_dim = hidden_dim,
            layers = layers_num,
            hidden_dim = 2 * hidden_dim,
            alpha = alpha,
            eta = eta,
            theta = theta   
        )
        # 最终输出层
        self.final_layer = nn.Linear(self.inter_dim * hidden_dim, seq_len * hidden_dim)

        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.deivce = None
        
        # 参数列表
        self.outer_params = [self.persistent_memory] + list(self.final_layer.parameters()) + list(self.att_layer.parameters())


    # 定义前向传播函数
    def forward(self, X, dc_memory):
        # X.shape = (batch_size, seq_len, hidden_dim)
        batch_size = X.shape[0]

        # 从neural memory模块中检索
        nmm_vals = self.nm_module.retrieve(X.view(-1, self.hidden_dim)).view(batch_size, -1, self.hidden_dim)
        # 拼接各个部分
        pm_expanded = self.persistent_memory.unsqueeze(0).expand(X.shape[0], -1, -1)
        X = torch.cat([dc_memory, pm_expanded, nmm_vals, X], dim = 1)

        # 通过注意力层处理
        X = self.silu(self.att_layer(X).view(-1, self.inter_dim * self.hidden_dim))
        X = self.final_layer(X).view(-1, self.hidden_dim)

        # 更新neural memory模块
        _, new_params = self.nm_module.update(X)
        y = functional_call(self.nm_module, X, new_params)

        # 最终输出
        return (X * self.sigmoid(y)).view(batch_size, self.seq_len, self.hidden_dim)
    

class GateTitan(nn.Module):
    pass

class Titan(nn.Module):

    def __init__(
            self, 
            input_dim,
            hidden_dim, 
            output_dim,
            context_window,
            pm_len, 
            dc_len,
            n_layers = 2,
            n_layers_nmm = 2, 
            alpha = 0.999, 
            eta = 0.8, 
            theta = 0.3):
        
        super().__init__()

        # 初始化各个维度和参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_window = context_window

        # transformer 的输入嵌入层
        self.embeding_layer = nn.Linear(input_dim, hidden_dim)

        # 堆叠多个memory层
        self.layers = nn.ModuleList([
            MACTitan(
                hidden_dim,
                context_window,
                pm_len,
                dc_len,
                layers_num = n_layers_nmm,
                alpha = alpha,
                eta = eta,
                theta = theta
            )
            for _ in range(n_layers)
        ])

        # 最终输出层
        self.final_layer = nn.Linear(context_window * hidden_dim, output_dim)
        self.silu = nn.SiLU()

        # 汇总所有外部参数，便于统一管理
        self.outer_params = list(self.embeding_layer.parameters()) + list(self.final_layer.parameters())
        for layer in self.layers:
            self.outer_params += layer.outer_params

        # 初始化DC层
        from config_loader import load_config
        from dynamic_cheatsheet import LLMCheatsheetMemory

        self.cfg = load_config("config.yaml")
        self.pathb = LLMCheatsheetMemory(self.cfg.path_b.__dict__, hidden_dim=self.hidden_dim)

        
    # 处理一个窗口的输入，输出(batch_size, output_dim)
    def process(self, X):
        batch_size = X.shape[0]
        # 输入嵌入
        X = self.embeding_layer(X.reshape(-1, self.input_dim)).view(batch_size, self.context_window, self.hidden_dim)
        
        # 2) DC 检索：得到 dc_mem (B, dc_len, hidden_dim)
        #    这里 query_text/window_text 先用占位，你后续可以换成真实文本摘要
        if hasattr(self, "dc") and self.cfg.dc.enabled and self.cfg.dc.mode != "off":
            query_text = "window_query"
            dc_mem, _ = self.dc.retrieve(
                query_text=query_text,
                batch_size=batch_size,
                device=X.device
            )
        else:
            # DC 关闭时：给全 0（形状必须对）
            dc_len = getattr(getattr(self.cfg, "dc", None), "embedding", None).dc_len if hasattr(self, "cfg") else 8
            dc_mem = torch.zeros(batch_size, dc_len, self.hidden_dim, device=X.device)
        
        # 依次通过所有MACTitan层，每层输出加残差
        for layer in self.layers:
            X = X + self.silu(layer(X, dc_mem))
        # 最终输出
        return self.final_layer(X.reshape(batch_size, -1))
    
    # 前向传播
    def forward(self, X):
        # x: (batch_size, N, input_dim)，N为序列长度
        # 输出: (batch_size, N, output_dim)
        res = torch.zeros((X.shape[0], X.shape[1], self.output_dim)).cuda()
        
        # 初始化输出
        X = np.permute_dims(sliding_window_view(X.cpu(), self.context_window, axis=1), (0,1,3,2))
        residual = X.shape[1] % self.context_window

        stz = torch.from_numpy(X[:,:-residual].reshape(X.shape[0], -1, self.context_window, self.context_window, self.input_dim)).cuda()
        for i in range(stz.shape[1]):
            slide = stz[:,i].reshape(-1, self.context_window, X.shape[-1])
            out = self.process(slide).reshape(-1, self.context_window, self.output_dim)
            res[:, (i+1)*self.context_window -1:(i+2)*self.context_window -1] = out
        # 处理不能整除的部分
        residual_part = torch.from_numpy(X[:,-residual:].reshape(-1, self.context_window, self.input_dim)).cuda()
        res_out = self.process(residual_part).reshape(-1, residual, self.output_dim)
        res[:, -residual:] = res_out

        return res