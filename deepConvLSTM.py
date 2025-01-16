import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DeepConvLSTM(nn.Module):
    def __init__(self, num_classes):
        super(DeepConvLSTM, self).__init__()
        
        # 1) Convolutional layers
        self.conv1 = nn.Conv1d(3,   64, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=0)
        
        # 2) LSTM layers
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        
        # 3) Dropout
        self.dropout = nn.Dropout(p=0.5)
        
        # 4) Classifier
        self.fc = nn.Linear(128, num_classes)
        
        # 5) Softmax final activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths):
        """
        x: (batch_size, max_windows, window_size=60, channels=3)
        lengths: list or 1D tensor with actual #windows for each sample
        """
        batch_size, max_windows, window_size, channels = x.shape
        # Example: (N, T_max=some, 60, 3)

        # -------------------------------------------------------------
        # (A) Merge (N, T_max) so we can run 1D conv across each window
        #     => shape: (N*T_max, window_size, channels)
        x = x.view(batch_size * max_windows, window_size, channels)

        # Permute => (N*T_max, channels=3, length=60) for Conv1D
        x = x.permute(0, 2, 1)  # => (N*T_max, 3, 60)

        # -------------------------------------------------------------
        # (B) Convolutional feature extraction
        x = nn.functional.relu(self.conv1(x))  # => (N*T_max, 64, L1)
        x = nn.functional.relu(self.conv2(x))  # => (N*T_max, 64, L2)
        x = nn.functional.relu(self.conv3(x))  # => (N*T_max, 64, L3)
        x = nn.functional.relu(self.conv4(x))  # => (N*T_max, 64, L4)

        # In the Ordóñez & Roggen paper, they ended with 64 features per “window.”  
        # We'll do global average pooling along the last dimension:
        x = x.mean(dim=2)  # => (N*T_max, 64)

        # -------------------------------------------------------------
        # (C) Un-merge => shape: (N, T_max, 64)
        x = x.view(batch_size, max_windows, 64)

        # -------------------------------------------------------------
        # (D) Pack sequences for LSTM1
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm1(packed_x)
        
        # **Unpack** after LSTM1, then apply dropout
        out1, _ = pad_packed_sequence(packed_out, batch_first=True)  # => (N, T_max, 128)
        out1 = self.dropout(out1)

        # **Re-pack** for LSTM2
        packed_out2 = pack_padded_sequence(out1, lengths, batch_first=True, enforce_sorted=False)
        packed_out2, _ = self.lstm2(packed_out2)

        # **Unpack** after LSTM2, then apply dropout
        out2, _ = pad_packed_sequence(packed_out2, batch_first=True)  # => (N, T_max, 128)
        out2 = self.dropout(out2)

        # -------------------------------------------------------------
        # (E) Gather the last valid output from each sequence
        #     lengths[i] is the #windows for sample i, so last valid idx = lengths[i]-1
        device = out2.device
        idx = (torch.tensor(lengths, device=device) - 1).long()  # shape [N]
        
        # Gather last time-step from each sequence
        # out2 is [N, T_max, 128], we pick out2[i, idx[i], :]
        # One easy way is with a loop or with "gather":
        idx = idx.view(-1, 1, 1).expand(-1, 1, out2.size(2))  # => shape [N,1,128]
        last_outputs = out2.gather(1, idx).squeeze(1)        # => shape [N,128]

        # **Optional dropout** after extracting last output
        last_outputs = self.dropout(last_outputs)

        # -------------------------------------------------------------
        # (F) Final Dense layer => apply dropout again
        x = self.fc(last_outputs)      # => (N, num_classes)
        x = self.dropout(x)            # dropout after dense
        x = self.softmax(x)            # => (N, num_classes)

        return x

class AccDataset(Dataset):
    def __init__(self, data, labels, lengths):
        self.data = data
        self.labels = labels
        self.lengths = lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.lengths[idx]