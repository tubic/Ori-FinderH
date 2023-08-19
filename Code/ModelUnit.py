import torch
import torch.nn as nn

# 12+12+36+48+144


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        Q = self.query(x).unsqueeze(1)
        K = self.key(x).unsqueeze(2)
        V = self.value(x).unsqueeze(2)
        att_weights = torch.matmul(Q, K.transpose(-1, 2))
        att_weights = att_weights / (self.hidden_size ** 0.5)
        att_weights = torch.softmax(att_weights, dim=-1)
        att_output = torch.matmul(att_weights, V).squeeze(2)        
        return att_output


class MainModel(nn.Module):
    def __init__(self, in_channels):
        super(MainModel, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, )),
            nn.ELU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, )),
            nn.ELU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, )),
            nn.ELU(),
        )

        self.down_sample_1 = nn.AvgPool1d(kernel_size=(3, ))
        self.down_sample_2 = nn.AvgPool1d(kernel_size=(3, ))
        self.down_sample_3 = nn.AvgPool1d(kernel_size=(3, ))

        self.self_attention_1 = SelfAttention(input_size=12+12+36+48+144, hidden_size=12+12+36+48+144)
        self.self_attention_2 = SelfAttention(input_size=166, hidden_size=166)
        self.self_attention_3 = SelfAttention(input_size=110, hidden_size=110)
        self.self_attention_4 = SelfAttention(input_size=72, hidden_size=72)

        self.output_block = nn.Sequential(
            nn.Linear(in_features=72, out_features=2),
            nn.Softmax(dim=-1)
        )

        self.norm_2 = nn.LayerNorm(166)
        self.norm_3 = nn.LayerNorm(110)
        self.norm_4 = nn.LayerNorm(72)


    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, x.shape[-1]))
        x = self.self_attention_1(x)
        x = self.conv_block_1(x)
        x = self.down_sample_1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)

        x = self.norm_2(x)
        x = self.self_attention_2(x)
        x = self.conv_block_2(x)
        x = self.down_sample_2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        
        x = self.norm_3(x)
        x = self.self_attention_3(x)
        x = self.conv_block_3(x)
        x = self.down_sample_3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)

        x = self.norm_4(x)
        x = self.self_attention_4(x)
        x = self.output_block(x)
        return torch.reshape(x, (x.shape[0], x.shape[-1]))