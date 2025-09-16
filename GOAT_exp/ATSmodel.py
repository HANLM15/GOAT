import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join

class Att(nn.Module):
    def __init__(self, hidden_size, attention_size, num_layers):
        super(Att, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.num_layers = num_layers

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * num_layers, attention_size))
        self.u_omega = nn.Parameter(torch.Tensor(attention_size))
        nn.init.constant_(self.w_omega, 0.001)
        nn.init.constant_(self.u_omega, 0.001)

    def forward(self, x, gru_output):
        # import pdb; pdb.set_trace()
        device = x.device
        # x[128, 98, 768];gruoutput128, 98, 256]
        mask = torch.sign(torch.abs(torch.sum(x, axis=-1))).to(device)#[128, 98]
        #self.w_omega [256, 4]
        attn_tanh = torch.tanh(torch.matmul(gru_output, self.w_omega))#[128, 98, 4]
        attn_hidden_layer = torch.matmul(attn_tanh, self.u_omega)#[128, 98]
        paddings = torch.ones_like(mask) * (-10e8)
        attn_hidden_layer = torch.where(torch.eq(mask, 0), paddings, attn_hidden_layer)
        alphas = F.softmax(attn_hidden_layer, 1)#[128, 98]
        attn_output = torch.sum(gru_output * torch.unsqueeze(alphas, -1), 1)#[128, 256]
        return attn_output
    
class TextEncoder(nn.Module):
    def __init__(self,cfg):
        super(TextEncoder,self).__init__()
        self.cfg = cfg
        text_feat_dim = cfg["model"]["text_feat_dim"]
        hidden_dim = cfg["model"]["hidden_dim"]
        num_classes = cfg["model"]["num_classes"]
        num_heads = cfg["model"]["num_heads"]
        self.num_layers = cfg["model"]["num_layers"]
        self.direction = cfg["model"]["direction"]
        self.hidden_size1 = cfg["model"]["hidden_size1"]
        self.attention_size = cfg["model"]["attention_size"]
        self.lstm1 = nn.LSTM(text_feat_dim, self.hidden_size1, self.num_layers, batch_first=True, bidirectional=True)
        self.Att1 = Att(self.hidden_size1, self.attention_size, self.num_layers)
    def forward(self,text_feat):
        device = text_feat.device 
        text_feat = text_feat.to(device)
        h1 = torch.zeros(self.num_layers * self.direction, text_feat.size(0),self.hidden_size1).to(device)
        c1 = torch.zeros(self.num_layers * self.direction, text_feat.size(0),self.hidden_size1).to(device)
        out1, _ = self.lstm1(text_feat, (h1,c1))
        outa1 = self.Att1(text_feat, out1)
        return outa1
    

class AudioEncoder(nn.Module):
    def __init__(self,cfg):
        super(AudioEncoder,self).__init__()
        self.cfg = cfg
        audio_feat_dim = cfg["model"]["audio_feat_dim"]
        hidden_dim = cfg["model"]["hidden_dim"]
        num_classes = cfg["model"]["num_classes"]
        num_heads = cfg["model"]["num_heads"]
        self.num_layers = cfg["model"]["num_layers"]
        self.direction = cfg["model"]["direction"]
        self.hidden_size2 = cfg["model"]["hidden_size2"]
        self.attention_size = cfg["model"]["attention_size"]
        self.lstm2 = nn.LSTM(audio_feat_dim, self.hidden_size2, self.num_layers, batch_first=True, bidirectional=True)
        self.Att2 = Att(self.hidden_size2, self.attention_size, self.num_layers)
    def forward(self,audio_feat):
        device = audio_feat.device 
        audio_feat = audio_feat.to(device)
        h2 = torch.zeros(self.num_layers * self.direction, audio_feat.size(0),self.hidden_size2).to(device)
        c2 = torch.zeros(self.num_layers * self.direction, audio_feat.size(0),self.hidden_size2).to(device)
        out2, _ = self.lstm2(audio_feat, (h2,c2))
        outa2 = self.Att2(audio_feat, out2)
        return outa2


class SharedHead(nn.Module):
    def __init__(self,cfg):
        super(SharedHead,self).__init__()
        self.cfg = cfg
        num_classes = cfg["model"]["num_classes"]
        self.num_layers = cfg["model"]["num_layers"]
        self.direction = cfg["model"]["direction"]
        self.hidden_size1 = cfg["model"]["hidden_size1"]
        self.hidden_size2 = cfg["model"]["hidden_size2"]
        self.st = nn.Sequential(
            nn.Linear(self.hidden_size1* 2, 64),
            nn.ReLU())
        self.c_fc = nn.Linear(64, num_classes)
        
    def forward(self,out):
        fc_out = self.st(out)
        output = self.c_fc(fc_out)
        # return output,self.ss(outa1), self.tt(outa2)
        return output,fc_out