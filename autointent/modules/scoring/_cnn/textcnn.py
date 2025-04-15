import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 n_classes, 
                 embed_dim=128, 
                 kernel_sizes=(3, 4, 5), 
                 num_filters=100, 
                 dropout=0.1,
                 padding_idx=0,
                 pretrained_embs=None
        ):
        super(TextCNN, self).__init__()
        
        if pretrained_embs != None:
            _, embed_dim = pretrained_embs.shape
            self.embedding = nn.Embedding.from_pretrained(pretrained_embs, freeze=True)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), n_classes)

    def forward(self, x):
        x = self.embedding(x)  
        x = x.permute(0, 2, 1)
        x = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs] 
        x = torch.cat(x, dim=1) 
        x = self.dropout(x)
        x = self.fc(x)

        return x