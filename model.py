import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

class Contracting_unit(nn.Module):
	def __init__(self,in_channels,out_channels):
		super(Contracting_unit,self).__init__()
		self.Conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels,
						kernel_size=3,stride=1,padding=1),
			nn.PReLU())
		self.Conv2=nn.Sequential(
			nn.Conv2d(out_channels,out_channels,
						kernel_size=3,stride=1,padding=1),
			nn.PReLU())
		self.Pool=nn.MaxPool2d(kernel_size=2,stride=2)

	def forward(self,x):
		x=self.Conv1(x)
		y=self.Conv2(x)
		x=self.Pool(y)
		return x,y

class Expansive_unit(nn.Module):
	def __init__(self,in_channels,out_channels):
		super(Expansive_unit,self).__init__()
		self.Conv1=nn.Sequential(
			nn.Conv2d(in_channels,out_channels,
						kernel_size=3,stride=1,padding=1),
			nn.PReLU())
		self.Conv2=nn.Sequential(
			nn.Conv2d(out_channels,out_channels,
						kernel_size=3,stride=1,padding=1),
			nn.PReLU())
		self.DC=nn.Sequential(
			nn.ConvTranspose2d(out_channels,
			out_channels,kernel_size=2,stride=2),
			nn.PReLU())

	def forward(self,x1,y):
		x=self.Conv1(x1)
		x=self.Conv2(x)
		x=self.DC(x)
		x=torch.cat((x,y),1)
		return x

class Fusion_Network(nn.Module):
	"""docstring for Fusion_Network"""
	def __init__(self, HS_channel=31, MS_channel=3):
		super(Fusion_Network, self).__init__()
		self.HS_channel=HS_channel
		self.MS_channel=MS_channel

		self.HS_double_conv1=nn.Sequential(
			nn.Conv2d(in_channels=HS_channel,out_channels=16,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU(),
			nn.Conv2d(in_channels=16,out_channels=32,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU())
		self.HS_down_conv1=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=32,
				kernel_size=3,stride=2,padding=1),
			nn.PReLU())

		self.MS_double_conv1=nn.Sequential(
			nn.Conv2d(in_channels=MS_channel,out_channels=16,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU(),
			nn.Conv2d(in_channels=16,out_channels=32,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU())
		self.MS_down_conv1=nn.Sequential(
			nn.Conv2d(in_channels=32,out_channels=32,
				kernel_size=3,stride=2,padding=1),
			nn.PReLU())

		self.double_conv2=nn.Sequential(
			nn.Conv2d(in_channels=64,out_channels=64,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU(),
			nn.Conv2d(in_channels=64,out_channels=128,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU())
		self.down_conv2=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=128,
				kernel_size=3,stride=2,padding=1),
			nn.PReLU())

		self.double_conv3=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=256,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU(),
			nn.Conv2d(in_channels=256,out_channels=256,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU())

		self.up_conv1=nn.Sequential(
			nn.ConvTranspose2d(in_channels=256,out_channels=128,
				kernel_size=2,stride=2,padding=0),
			nn.PReLU())

		self.double_conv4=nn.Sequential(
			nn.Conv2d(in_channels=256,out_channels=128,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU(),
			nn.Conv2d(in_channels=128,out_channels=128,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU())

		self.up_conv2=nn.Sequential(
			nn.ConvTranspose2d(in_channels=128,out_channels=64,
				kernel_size=2,stride=2,padding=0),
			nn.PReLU())

		self.double_conv5=nn.Sequential(
			nn.Conv2d(in_channels=128,out_channels=64,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU(),
			nn.Conv2d(in_channels=64,out_channels=64,
				kernel_size=3,stride=1,padding=1),
			nn.PReLU())

		self.conv=nn.Sequential(
			nn.Conv2d(in_channels=64,out_channels=HS_channel,
				kernel_size=3,stride=1,padding=1),
			nn.Tanh())

	def forward(self,HS,MS):
		HS=self.HS_double_conv1(HS)
		x1=self.HS_down_conv1(HS)

		MS=self.MS_double_conv1(MS)
		x2=self.MS_down_conv1(MS)

		x=torch.cat((x1,x2),1)
		y=self.double_conv2(x)
		x=self.down_conv2(y)
		x=self.double_conv3(x)
		x=self.up_conv1(x)

		x=torch.cat((x,y),1)
		x=self.double_conv4(x)
		x=self.up_conv2(x)

		x=torch.cat((x,MS,HS),1)
		x=self.double_conv5(x)
		x=self.conv(x)

		return x

class Attention(nn.Module):
	def __init__(self,MS_channel=3):
		super(Attention,self).__init__()
		self.MS_channel=MS_channel

		self.Unit1=Contracting_unit(MS_channel,32)
		self.Unit2=Contracting_unit(32,64)
		self.Unit3=Contracting_unit(64,128)
		self.Unit4=Contracting_unit(128,256)

		self.Unit5=Expansive_unit(256,512)
		self.Unit6=Expansive_unit(768,256)
		self.Unit7=Expansive_unit(384,128)
		self.Unit8=Expansive_unit(192,64)

		self.conv=nn.Sequential(
			nn.Conv2d(in_channels=96,out_channels=1,
				kernel_size=3,stride=1,padding=1),
			nn.Sigmoid())

	def forward(self,x):

		x,y1=self.Unit1(x)
		x,y2=self.Unit2(x)
		x,y3=self.Unit3(x)
		feature,y4=self.Unit4(x)

		x=self.Unit5(feature,y4)
		x=self.Unit6(x,y3)
		x=self.Unit7(x,y2)
		x=self.Unit8(x,y1)

		x=self.conv(x)
		return x

class Deep_Attention_Network(nn.Module):
	"""docstring for Deep_Attention_Network"""
	def __init__(self, if_attention=True):
		super(Deep_Attention_Network, self).__init__()
		self.if_attention=if_attention

		self.Fusion_Network=Fusion_Network()
		self.Attention=Attention()

	def forward(self,HS,MS):

		fusion=self.Fusion_Network(HS,MS)
		mask=self.Attention(MS)

		if self.if_attention:
			return fusion*mask+fusion
		else:
			return fusion
		
if __name__=='__main__':
	HS=torch.rand((1,31,256,256))
	MS=torch.rand((1,3,256,256))
	model=Deep_Attention_Network()
	x=model(HS,MS)
	print(x.shape)