import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0):
        super(DecoderRNN, self).__init__()
       
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers==1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        captions = captions[:,:-1]
        batch_size = features.shape[0]
        features = features.unsqueeze(1)
        embedded_captions = self.embedding(captions)
        lstm_in = torch.cat((features, embedded_captions), 1)
        lstm_out, _ = self.lstm(lstm_in)
        captions_out = self.fc(lstm_out)
        
        return captions_out 

    def sample(self, inputs, states=None, max_len=20, end_word_idx = 1):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        captions = []
        
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            fc_out = self.fc(lstm_out)
            prediction = fc_out.argmax(dim=2)
            caption = prediction.item() 
            captions.append(caption)
            inputs = self.embedding(prediction)
            
            if len(captions) >= max_len or caption == end_word_idx:
                break
            
        return captions
            
            
            
        