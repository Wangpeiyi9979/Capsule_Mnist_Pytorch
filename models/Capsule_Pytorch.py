import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(x, axis=-1):
    """
    功能: 实现Capsule中的压缩激活
    输入:
        - x(tensor)[B, num_capsule, dim_capsule]
        - axis: 要压缩的维度，标准的Capsule为-1
    输出:
        - out(tensor)[B, num_capsule, dim_capsule]
    """
    x_squared_norm = torch.sum(torch.pow(x, 2), axis, True) + 1e-7  #(B, num_capsule, 1)
    scale = torch.sqrt(x_squared_norm) / (0.5 + x_squared_norm)
    return scale * x

class Capsule(nn.Module):
    def __init__(self, input_dim, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        """
        参数:
            - ID: input_dim(int): 上一层胶囊输入的向量维度
            - NC: num_capsule(int): 下一层输出的胶囊数
            - DC: dim_capsule(int): 下一层输出中的每个胶囊所含向量维数
            - routings(int): 执行几次动态路由过程，即聚类迭代次数
            - share_weights(bool): 是否共享层
            - activation(str):   采用何种非线性函数，默认为squash
        """

        super(Capsule, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights

        if activation == 'squash':
            self.activation = squash
        """
        这里可以添加一些其他的激活函数，功能是: 将向量其模长压缩到0—1之间
        """
        if self.share_weights:
            self.W = nn.Linear(self.input_dim, self.num_capsule*self.dim_capsule, bias=False)
        else:
            pass

        self.init_Weight()

    def init_Weight(self):
        nn.init.xavier_normal_(self.W.weight)

    def forward(self, u_vecs):
        """
        功能: 实现Capsule的聚类过程
        参数:
            - u_vecs(tensor)[B, IN, ID]: IN为上一层胶囊数，如果上一层接RNN，则ID为其hidden_dim, 若上一层接CNN, 则ID为其filter_num
        输出:
            - out(tensor)[B, ON, DC]
        """
        if self.share_weights:
            u_hat_vecs = self.W(u_vecs)          # (B, IN, ON*DC)
        batch_size = u_hat_vecs.size(0)
        input_num_capsule = u_hat_vecs.size(1)

        u_hat_vecs = u_hat_vecs.view(batch_size, input_num_capsule, self.num_capsule, self.dim_capsule)  # (B, IN, ON, DC)
        u_hat_vecs = u_hat_vecs.permute(0, 2, 1, 3)             # (B, ON, IN, DC)
        b = torch.zeros_like(u_hat_vecs[:, :, :, 0])            # (B, ON, IN)

        for i in range(self.routings):
            c = torch.softmax(b, 1)
            o = torch.matmul(c.unsqueeze(2), u_hat_vecs).squeeze(2)        # (B, ON, DC)
            if i < self.routings - 1:
                o = F.normalize(o, 2, -1)                                          # (B, ON, DC)
                b = torch.matmul(u_hat_vecs, o.unsqueeze(-1)).squeeze(-1)          # (B, ON, IN)
        return self.activation(o)

