import torch
import torch.nn as nn

import torch.nn.functional as F
import smile as sm
from smile import flags, logging


class CBE(nn.Module):
    def __init__(self, config):
        super(CBE, self).__init__()

        self.conv_bot = nn.Sequential(nn.Conv1d(in_channels=config.feature_size,
                                                out_channels=config.node_size,
                                                kernel_size=config.bot_window_size,
                                                padding=int((config.bot_window_size-1)/2)),
                                      #   nn.BatchNorm1d(num_features=config.node_size),
                                      nn.Dropout(config.dropout_rate),
                                      nn.ReLU()
                                      )
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.node_size,
                                    out_channels=config.node_size,
                                    kernel_size=h,
                                    padding=int((h-1)/2)),
                          #  nn.BatchNorm1d(num_features=config.node_size),
                          nn.Dropout(config.dropout_rate),
                          nn.ReLU()
                          )
            for h in config.window_sizes
        ])
        self.conv_top = nn.Sequential(nn.Conv1d(in_channels=config.node_size,
                                                out_channels=config.node_size,
                                                kernel_size=config.top_window_size,
                                                padding=int((config.top_window_size-1)/2)),
                                      #   nn.BatchNorm1d(num_features=config.node_size),
                                      nn.Dropout(config.output_dropout_rate),
                                      nn.ReLU()
                                      )

        self.bilstm = nn.LSTM(
            input_size=config.feature_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.blstm_layer_num,
            batch_first=True,
            dropout=config.lstm_dropout_rate,
            bidirectional=True
        )

        self.fc1 = nn.Sequential(
            nn.Sigmoid(),
            # nn.Dropout(config.fc1_dropout_rate),
            nn.Linear(config.node_size +
                      config.lstm_hidden_size*2, config.fc1_dim),
            nn.Dropout(config.fc1_dropout_rate),
            nn.ReLU()
        )

        self.device = config.device
        self.lstm_hidden_size = config.lstm_hidden_size
        self.feature_size = config.feature_size
        self.batch_size = config.batch_size
        self.blstm_layer_num = config.blstm_layer_num

    def init_hidden(self, batch_size):
        return (torch.zeros(self.blstm_layer_num * 2, batch_size, self.lstm_hidden_size).to(self.device),
                torch.zeros(self.blstm_layer_num * 2, batch_size, self.lstm_hidden_size).to(self.device))

    def forward(self, x):
        # x.size() = [32,42,700]
        cnnout = self.conv_bot(x)  # [32,100,700]
        for conv in self.convs:
            cnnout = conv(cnnout)
        cnnout = self.conv_top(cnnout)  # [32,100,700]
        cnnout = cnnout.permute(0, 2, 1)  # [32,700,100]

        x = x.permute(0, 2, 1)  # [32,700,42]
        hidden_states = self.init_hidden(x.shape[0])
        bilstm_out, hidden_states = self.bilstm(
            x, hidden_states)  # bilstm_out=[32,700,1024]

        out = torch.cat((cnnout, bilstm_out), 2)
        out = self.fc1(out)
        out = out.permute(0, 2, 1)  # (32,21,700)

        return out
