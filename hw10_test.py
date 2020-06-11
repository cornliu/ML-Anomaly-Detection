import numpy as np
import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
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

batch_size = 128

if os.path.split(sys.argv[2])[1] == "baseline.pth":
	print("Model type: cnn")
	model_type = 'cnn'
elif os.path.split(sys.argv[2])[1] == "best.pth":
	print("Model type: fcn")
	model_type = 'fcn'
else:
	print("Unknown model type of {}. Assuming fcn model.".format(os.path.split(sys.argv[2])[1]))
	model_type = 'fcn'

test = np.load(sys.argv[1], allow_pickle=True)

if model_type == 'fcn':
    y = test.reshape(len(test), -1)
else:
    y = test
    
data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

model = torch.load(sys.argv[2], map_location='cuda')

model.eval()
reconstructed = list()
for i, data in enumerate(test_dataloader):
    if model_type == 'cnn':
        img = data[0].transpose(3, 1).cuda()
    else:
        img = data[0].cuda()
    output = model(img)
    if model_type == 'cnn':
        output = output.transpose(3, 1)
    elif model_type == 'vae':
        output = output[0]
    reconstructed.append(output.cpu().detach().numpy())

reconstructed = np.concatenate(reconstructed, axis=0)
anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
y_pred = anomality
with open(sys.argv[3], 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))