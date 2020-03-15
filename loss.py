import torch
from data import *
import torch.nn as nn
from torch.nn import functional as F

class l1_loss(nn.Module):
	"""docstring for l1_loss"""
	def __init__(self):
		super(l1_loss, self).__init__()

	def forward(self,y,z):
		return torch.abs(y-z).mean()
		

if __name__=='__main__':
	criterion=l1_loss()
	input=torch.zeros((1,1,320,320))
	output=torch.ones((1,1,320,320))
	loss=criterion(input,output)
	print(loss)