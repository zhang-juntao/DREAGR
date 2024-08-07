import torch
import torch.nn as nn

class MeanPoolAggregator(nn.Module):
    """ Group Preference Aggregator implemented as mean pooling over group member embeddings """

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(MeanPoolAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(drop_ratio)
        )
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=True):
        if mlp:
            h = torch.tanh(self.mlp(x))
        else:
            h = x
        if mask is None:
            return torch.mean(h, dim=1)
        else:
            mask = torch.exp(mask)
            res = torch.sum(h * mask.unsqueeze(2), dim=1) / mask.sum(1).unsqueeze(1)
            return res


class AttentionAggregator(nn.Module):
    """ Group Preference Aggregator implemented as attention over group member embeddings """

    def __init__(self, input_dim, output_dim, drop_ratio=0):
        super(AttentionAggregator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.Sigmoid(),
            nn.Dropout(drop_ratio)
        )

        self.attention = nn.Linear(output_dim, 1)
        self.drop = nn.Dropout(drop_ratio)
        nn.init.xavier_uniform_(self.mlp[0].weight)
        if self.mlp[0].bias is not None:
            self.mlp[0].bias.data.fill_(0.0)

    def forward(self, x, mask, mlp=True):
        if mlp:
            h = torch.tanh(self.mlp(x))
            #t = torch.tanh(x)
        else:
            h = x

        attention_out = torch.tanh(self.attention(h))
        if mask is None:
            weight = torch.softmax(attention_out, dim=1)
        else:
            weight = torch.softmax(attention_out + mask.unsqueeze(2), dim=1)
        ret = torch.matmul(h.transpose(2, 1), weight).squeeze(2)
        return ret
