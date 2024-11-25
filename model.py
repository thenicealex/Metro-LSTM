import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.input_size = cfg["input_size"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]
        self.bidirectional = cfg["bidirectional"]

        self.conv1d_in_channel = cfg["conv1d"]["in_channels"]
        self.conv1d_out_channel = cfg["conv1d"]["out_channels"]
        self.conv1d_kernel_size = cfg["conv1d"]["kernel_size"]
        self.hidden_size = cfg["hidden_size"]

        self.cov1 = nn.Conv1d(
            self.conv1d_in_channel, self.conv1d_out_channel, self.conv1d_kernel_size
        )
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        # self.dropout = nn.Dropout(self.dropout)
        self.fc_hidden_size = self.hidden_size
        if self.bidirectional:
            self.fc_hidden_size *= 2
        self.fc = nn.Linear(self.fc_hidden_size, 1)

    def forward(self, x):
        x = self.cov1(x)
        if self.bidirectional:
            num_layers = self.num_layers * 2
        else:
            num_layers = self.num_layers
        h0 = torch.zeros(num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # out = self.dropout(out[:, -1, :])
        out = self.fc(out[:, -1, :])
        return out


class MultiLSTM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.input_size = cfg["input_size"]
        self.num_layers = cfg["num_layers"]
        self.dropout = cfg["dropout"]
        self.bidirectional = cfg["bidirectional"]

        self.conv1d_in_channel = cfg["conv1d"]["in_channels"]
        self.conv1d_out_channel = cfg["conv1d"]["out_channels"]
        self.conv1d_kernel_size = cfg["conv1d"]["kernel_size"]
        self.hidden_size = cfg["hidden_size"]

        # self.cov1d = nn.Conv1d(
        #     self.conv1d_in_channel, self.conv1d_out_channel, self.conv1d_kernel_size
        # )
        self.lstm = nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        # self.dropout = nn.Dropout(self.dropout)
        self.fc_hidden_size = self.hidden_size
        if self.bidirectional:
            self.fc_hidden_size *= 2
        self.fc = nn.Linear(self.fc_hidden_size, 1)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # x = self.cov1d(x)
        # x = x.permute(0, 2, 1)

        num_layers = self.num_layers * 2 if self.bidirectional else self.num_layers
        h0 = torch.zeros(num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # out = self.dropout(out[:, -1, :])
        out = self.fc(out[:, -1, :])
        return out
