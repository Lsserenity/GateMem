import torch
from torch import nn, optim
from torch.nn.functional import normalize
from torch.func import functional_call

# 定义类似titans的NeuralMemory模块
class NeuralMemory(nn.Module):

    def __init__(self, 
                 input_dim = 16,    # 输入维度
                 layers = 2,        # MLP层数
                 hidden_dim = 32,   # 隐藏层维度
                 alpha = 0.999,     
                 eta = 0.60, 
                 theta = 0.05):
        
        super().__init__()
        
        self.layers = None
        # 定义MLP层结构
        if layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim)])
        else:
            self.layers = nn.ModuleList([])
            for i in range (layers - 1):
                self.layers.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.SiLU()
                ))
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.SiLU()
            ))

        # 定义key和value的参数层
        self.K = nn.Linear(input_dim, input_dim, bias = False)
        self.V = nn.Linear(input_dim, input_dim, bias = False)
        self.Q = nn.Linear(input_dim, input_dim, bias = False)

        # 初始化key和value层的权重
        torch.nn.init.xavier_uniform_(self.K.weight)
        torch.nn.init.xavier_uniform_(self.V.weight)
        torch.nn.init.xavier_uniform_(self.Q.weight)

        # 定义参数更新相关的超参数
        self.alpha = alpha
        self.eta = eta
        self.theta = theta

        # 存储惊讶度
        self.surprise = {}
        self.silu = nn.SiLU()

        # 记录新参数
        self.new_params = None
    
    # 定义前向传播函数
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    # 定义记忆查询函数
    def retrieve(self, X, new_params = None):
        if new_params is None:
            params = dict(self.named_parameters())
        else:
            params = new_params
        return functional_call(self, params, self.silu(normalize(self.Q(X))))
    
    # # 定义记忆更新函数
    # def update(self, X):
    #     x = X.detach()

    #     # 计算k和v
    #     k = normalize(self.silu(self.K(x)))
    #     v = self.silu(self.V(x))

    #     for layer in self.layers:
    #         x = layer(x)
        
    #     # 计算loss和梯度
    #     loss = ((k - v)**2).mean()
    #     grads = torch.autograd.grad(loss, self.parameters())

    #     # 更新mlp的参数
    #     update_params = {}
    #     for (name, param), grad in zip(self.named_parameters(), grads):
    #         if name not in self.surprise:
    #             self.surprise[name] = torch.zeros_like(grad)
    #         self.surprise[name] = self.eta * self.surprise[name] - self.theta * grad
    #         # 只更新mlp的参数
    #         root = name.split('.')[0]
    #         if root not in ['K', 'V', 'Q']:
    #             updated_param = self.alpha * param.data + self.surprise[name]
    #             update_params[name] = updated_param
    #         else:
    #             update_params[name] = param.data
        
    #     self.new_params = update_params
    def update(self, X):
        """
        Fast update (Titans-style):
        - Do NOT update module slow weights in-place
        - Only update self.new_params (fast weights) for layers.*
        """
        x = X.detach()

        # target: v = V(x)  (作为写入目标，不要求可微到 V)
        with torch.no_grad():
            v = self.silu(self.V(x))  # (N, C)

        # query: q = Q(x)
        q = normalize(self.silu(self.Q(x)))  # (N, C)

        # 当前使用的参数：如果已有 fast weights 用 fast，否则用 slow
        params = dict(self.named_parameters()) if self.new_params is None else self.new_params

        # forward through MLP layers using current params (必须依赖 layers.*)
        y = functional_call(self, params, q)  # self.forward() 只走 self.layers
        loss = ((y - v) ** 2).mean()

        # 只更新 layers.*（不更新 K/V/Q）
        layer_names = [f"layers.{n}" for n, _ in self.layers.named_parameters()]
        learnable = [params[n] for n in layer_names]

        grads = torch.autograd.grad(loss, learnable, allow_unused=False)

        # 初始化 surprise 缓冲
        if self.surprise is None:
            self.surprise = {}

        # 写回到 new_params（fast weights）
        update_params = dict(params)  # 从当前 params 复制一份开始

        for name, g in zip(layer_names, grads):
            if name not in self.surprise:
                self.surprise[name] = torch.zeros_like(g)

            # 你原来的动量/惊讶项规则（保留）
            self.surprise[name] = self.eta * self.surprise[name] - self.theta * g

            # fast weight update（保留你的 alpha）
            update_params[name] = self.alpha * params[name].detach() + self.surprise[name].detach()

        self.new_params = update_params
