import numpy as np
import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)

class fcn_autoencoder(nn.Module):
	def __init__(self):
		super(fcn_autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(32 * 32 * 3, 128),
			nn.ReLU(True),
			nn.Linear(128, 64),
			nn.ReLU(True),
			nn.Linear(64, 12),
			nn.ReLU(True),
			nn.Linear(12, 3)
			)
		self.decoder = nn.Sequential(
			nn.Linear(3, 12),
			nn.ReLU(True),
			nn.Linear(12, 64),
			nn.ReLU(True),
			nn.Linear(64, 128),
			nn.ReLU(True),
			nn.Linear(128, 32 * 32 * 3),
			nn.Tanh()
			)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

class conv_autoencoder(nn.Module):
	def __init__(self):
		super(conv_autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
			nn.ReLU(),
			nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
			nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
			nn.ReLU(),
			# nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
			# nn.ReLU(),
		)
		self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            # nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
			nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
			nn.ReLU(),
			nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
			nn.Tanh(),
		)

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

train = np.load(sys.argv[1], allow_pickle=True)

if os.path.split(sys.argv[2])[1] == "baseline.pth":
	print("Model type: cnn")
	model_type = 'cnn'
elif os.path.split(sys.argv[2])[1] == "best.pth":
	print("Model type: fcn")
	model_type = 'fcn'
else:
	print("Unknown model type of {}. Assuming fcn model.".format(os.path.split(sys.argv[2])[1]))
	model_type = 'fcn'

num_epochs = 6
batch_size = 128
learning_rate = 1e-5

x = train
if model_type == 'fcn':
	x = x.reshape(len(x), -1)
	
data = torch.tensor(x, dtype=torch.float)
train_dataset = TensorDataset(data)
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder()}
model = model_classes[model_type].cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(
	model.parameters(), lr=learning_rate)

best_loss = np.inf
model.train()
for epoch in range(num_epochs):
	for data in train_dataloader:
		if model_type == 'cnn':
			img = data[0].transpose(3, 1).cuda()
		else:
			img = data[0].cuda()
		output = model(img)
		loss = criterion(output, img)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if loss.item() < best_loss:
			best_loss = loss.item()
			torch.save(model, sys.argv[2])
	print('epoch [{}/{}], loss:{:.4f}'
			.format(epoch + 1, num_epochs, loss.item()))