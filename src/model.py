from torch import nn
import math
import torch

class FM(nn.Module):

    def __init__(self, args, user_num, item_num) -> None:
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dim = args.dim

        self.user_emb = nn.Embedding(user_num, self.dim)
        self.item_emb = nn.Embedding(item_num, self.dim)

        self.feat_dim = 2*self.dim

        # The one degree part in FM is implemented by Linear.
        self.oneDeg = nn.Linear(self.feat_dim, 1) # w_0 + \sum{w_i x_i}

        k = args.k
        self.twoDeg_v = nn.parameter.Parameter(torch.zeros(size=(self.feat_dim, k))) # eq.1 V

        self.loss_function = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1 / math.sqrt(self.dim)
        for weight in self.parameters():
            nn.init.normal_(weight.data,0,stdv)

    def forward(self, data):

        user_emb = self.user_emb(data['user'])
        item_emb = self.item_emb(data['item'])

        x = torch.concat([user_emb, item_emb],-1)
        # For simplicity,  only embedding is used. You can exceed other features by concatenating them there.

        ans = self.oneDeg(x).squeeze()

        v = torch.matmul(self.twoDeg_v, self.twoDeg_v.T)
        for i in range(self.feat_dim):
            for j in range(i+1, self.feat_dim):
                ans += (x[:,i] * x[:,j] * v[i,j])
        return ans