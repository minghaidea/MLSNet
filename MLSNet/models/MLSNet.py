import torch
import torch.nn as nn
import models.STVit as SA


class MLSNet(nn.Module):

    def __init__(self):
        super(MLSNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_to_8 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 1)) # conv to channel 8 -> STViT
        self.dropout = nn.Dropout(0.2)
        self.ELU = nn.ELU(inplace=True)
        self.BN = nn.BatchNorm2d(num_features=128)
        # **********************************************************
        self.Conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )
        self.Conv3 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(7, 7), padding=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.max_pooling_seq1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
        self.dropout_seq = nn.Dropout(0.2)
        self.convolution_seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=64, kernel_size=(12, 14), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )
        self.lstm_seq = nn.LSTM(64, 256, bidirectional=False, batch_first=True)
        self.max_pooling_seq2 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))
        self.convolution_seq_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 3), stride=(1, 1)),
        )
        # ***********************************************************
        self.STVit = SA.StokenAttention(8, stoken_size=[8, 8]).to(self.device)
        self.convolution_shape_1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=128, kernel_size=(5, 16), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 2))
        self.lstm = nn.LSTM(42, 21, 6, bidirectional=True, batch_first=True, dropout=0.2)  # Bi-LSTM
        self.convolution_shape_2 = nn.Sequential(
            nn.BatchNorm2d(num_features=128),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 3), stride=(1, 1)),
        )
        # ***********************************************************
        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def execute(self, seq, shape):
        seq = seq.float()
        shape = shape.float()
        seq = seq.unsqueeze(1)
        shape = shape.unsqueeze(1)

        seq_conv1 = self.Conv1(seq)
        seq_conv2 = self.Conv2(seq)
        seq_conv3 = self.Conv3(seq)
        seq = torch.cat((seq_conv1, seq_conv2, seq_conv3), dim=1) # multi fusion in channel
        seq = self.max_pooling_seq1(seq)
        seq = self.dropout_seq(seq)
        seq = self.convolution_seq_1(seq)
        seq = seq.squeeze(2)
        seq, _ = self.lstm_seq(seq.permute(0, 2, 1))
        seq = seq.permute(0, 2, 1)
        seq = seq.unsqueeze(2)
        seq = self.max_pooling_seq2(seq)
        seq = self.convolution_seq_2(seq)

        shape = self.conv_to_8(shape)
        shape = self.STVit(shape)
        shape = self.convolution_shape_1(shape)
        shape = self.max_pooling_1(shape)
        shape = shape.squeeze(2)
        shape, _ = self.lstm(shape)
        shape = shape.unsqueeze(2)
        shape = self.convolution_shape_2(shape)

        return self.output(torch.cat((shape, seq), dim=1))

    def forward(self, seq, shape):
        return self.execute(seq, shape)