import torch
import torch.nn as nn
from .Embedding import Smiles_embedding

class BERT_base(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.bert = model
		self.linear = output_layer
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(x)
		return x

class Smiles_BERT(nn.Module):
	def __init__(self, vocab_size, max_len=256, feature_dim=1024, nhead=4, feedforward_dim=1024, nlayers=6, adj=False, dropout_rate=0):
		super(Smiles_BERT, self).__init__()
		self.embedding = Smiles_embedding(vocab_size, feature_dim, max_len, adj=adj)
		trans_layer = nn.TransformerEncoderLayer(feature_dim, nhead, feedforward_dim, activation='gelu', dropout=dropout_rate)
		self.transformer_encoder = nn.TransformerEncoder(trans_layer, nlayers)
		
	def forward(self, src, pos_num, adj_mask=None, adj_mat=None):
		# True -> masking on zero-padding. False -> do nothing
		mask = (src == 0)
		mask = mask.type(torch.bool)

		x = self.embedding(src, pos_num, adj_mask, adj_mat)
		x = self.transformer_encoder(x.transpose(1,0), src_key_padding_mask=mask)
		x = x.transpose(1,0)
		return x

