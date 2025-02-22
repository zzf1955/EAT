import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb

class MLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        super(MLP, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim)
        )
        hidden_layer = []
        input_dim = state_dim + action_dim + t_dim
        for dim in hidden_dim:
            hidden_layer.append(nn.Linear(input_dim,dim))
            hidden_layer.append(_act())
            input_dim = dim
        hidden_layer.append(nn.Linear(input_dim,action_dim))
        self.mid_layer = nn.Sequential(*hidden_layer)
        self.final_layer = nn.Tanh()

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        state = state.reshape(state.size(0), -1)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SimpleSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = 1.0 / (embed_dim ** 0.5)

    def forward(self, x):
        # x: [batch_size, num_nodes, embed_dim]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_nodes, num_nodes]
        attention_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)  # [batch_size, num_nodes, embed_dim]
        return out

class AttentionMLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish',
        state_dim_per_node=3,  # 每个节点的特征维度
    ):
        super(AttentionMLP, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU

        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim)
        )

        # 状态的注意力层
        self.state_attention = SimpleSelfAttention(state_dim_per_node)
        # 如果您更倾向于使用内置的多头注意力层，也可以使用以下代码
        # self.state_attention = nn.MultiheadAttention(embed_dim=state_dim_per_node, num_heads=4, batch_first=True)

        # 确保 hidden_dim 是一个列表
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]

        # 构建隐藏层
        hidden_layer = []
        input_dim = state_dim + action_dim + t_dim
        for dim in hidden_dim:
            hidden_layer.append(nn.Linear(input_dim, dim))
            hidden_layer.append(_act())
            input_dim = dim
        hidden_layer.append(nn.Linear(input_dim, action_dim))
        self.mid_layer = nn.Sequential(*hidden_layer)

        self.final_layer = nn.Tanh()

    def forward(self, x, time, state):
        # 处理时间嵌入
        t = self.time_mlp(time)

        # 处理状态，应用注意力层
        # 假设 state 的形状是 [batch_size, num_nodes, state_dim_per_node]
        state = self.state_attention(state)  # 输出形状仍为 [batch_size, num_nodes, state_dim_per_node]
        state = state.reshape(state.size(0), -1)  # 展平成 [batch_size, num_nodes * state_dim_per_node]

        # 合并 x, t, state
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)

class DenseMLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=[256, 256],  # 隐藏层的维度可以是列表，定义每一层的神经元数
        t_dim=16,
        activation='mish'
    ):
        super(DenseMLP, self).__init__()
        
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        # 时间嵌入层
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim)
        )

        # 初始化输入维度
        self.input_dim = state_dim + action_dim + t_dim

        # 构建带有密集连接的隐藏层
        self.hidden_layers = nn.ModuleList()  # 用ModuleList存储所有层
        self.activations = nn.ModuleList()    # 用ModuleList存储每层的激活函数

        # 构建每个密集连接的层
        for dim in hidden_dim:
            self.hidden_layers.append(nn.Linear(self.input_dim, dim))
            self.activations.append(_act())
            # 由于密集连接，每一层的输入维度会增加
            self.input_dim += dim  # 下一个层的输入包含之前的所有输出

        # 最终输出层
        self.output_layer = nn.Linear(self.input_dim, action_dim)
        self.final_layer = nn.Tanh()

    def forward(self, x, time, state):
        # 时间嵌入
        t = self.time_mlp(time)
        state = state.reshape(state.size(0), -1)
        x = torch.cat([x, t, state], dim=1)

        # 遍历每个密集连接的层
        outputs = [x]  # 存储每层的输出以便密集连接
        for layer, activation in zip(self.hidden_layers, self.activations):
            # 将之前的所有输出连接到当前层输入
            x = torch.cat(outputs, dim=1)
            # 计算当前层的输出并激活
            x = activation(layer(x))
            outputs.append(x)  # 将当前层的输出加入到outputs列表

        # 最终输出层
        x = torch.cat(outputs, dim=1)  # 最终的输入包含所有层的输出
        x = self.output_layer(x)
        return self.final_layer(x)

class StateEmb(nn.Module):
    def __init__(
        self,
        state_dims,
        output_dim,
        hidden_dim=[256, 256],
        activation='mish',
        device = None
    ):
        super(StateEmb, self).__init__()
        
        # 卷积层
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, state_dims[1]))
        self.device = device
        # 激活函数
        if activation == 'mish':
            _act = nn.Mish()
        elif activation == 'relu':
            _act = nn.ReLU()

        # MLP 层
        layers = []
        input_dim = state_dims[0]
        for h_dim in hidden_dim:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(_act)
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, output_dim))  # 最后一层输出到 output_dim
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        # 将卷积应用于输入数据
        state = state.unsqueeze(1)
        x = self.conv(state)  # 经过卷积后，形状为 [batch_size, 1, height, 1]
        
        # 去掉多余的维度，并调整为 [batch_size, height] 形状
        x = x.squeeze(1).squeeze(-1)  # 去掉 channel 和 width 维度，变为 [batch_size, height]
        
        # MLP 层
        #x = self.mlp(x)  # 经过 MLP 层，输出为 [batch_size, output_dim]
        
        return x

class DoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=[256, 256],
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU

        # Define input dimension
        input_dim = state_dim + int(action_dim)

        # Build q1_net dynamically based on hidden_dims
        layers_q1 = []
        for h_dim in hidden_dim:
            layers_q1.append(nn.Linear(input_dim, h_dim))
            layers_q1.append(_act())
            input_dim = h_dim
        layers_q1.append(nn.Linear(input_dim, 1))
        self.q1_net = nn.Sequential(*layers_q1)

        # Reset input dimension for q2_net
        input_dim = state_dim + int(action_dim)

        # Build q2_net dynamically based on hidden_dims
        layers_q2 = []
        for h_dim in hidden_dim:
            layers_q2.append(nn.Linear(input_dim, h_dim))
            layers_q2.append(_act())
            input_dim = h_dim
        layers_q2.append(nn.Linear(input_dim, 1))
        self.q2_net = nn.Sequential(*layers_q2)

    def forward(self, state, action):
        # Concatenate state and action
        sa = torch.cat([state, action], dim=-1)

        # Compute two Q-values
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)

        return q1, q2

    def q_min(self, state, action):
        # Compute the minimum of the two Q-values
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class DoubleCritic_old(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU

        self.q1_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        obs = obs.reshape(obs.size(0), -1)
        return self.q1_net(obs), self.q2_net(obs)

    def q_min(self, obs):
        obs = obs.reshape(obs.size(0), -1)
        return torch.min(*self.forward(obs))
