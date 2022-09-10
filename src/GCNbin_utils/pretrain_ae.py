import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset




class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_dec_1, n_dec_2,
                 n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
  #      self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_2, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
    #    self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_2, n_input)
        #构建自编码器

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
    #    enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h2)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
      #  dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h2)


        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))





def adjust_learning_rate(optimizer, epoch):
    lr = 0.01 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def pretrain_ae(h1 , h2 , z, output_path, prefix, n_input, n_samples):
    model = AE(
        n_enc_1=h1,
        n_enc_2=h2,
        n_dec_1=h2,
        n_dec_2=h1,
        n_input=n_input,
        n_z=z,)
    x1 = np.loadtxt(output_path + prefix +"_normalized_contig_tetramers.txt", dtype=float)
    if n_samples != 0:
       x2 = np.loadtxt(output_path + prefix+"_normalized_coverages.txt", dtype=float)

       if n_samples==1:
           mi=min(x2)
           mami=max(x2)-mi
           x2=(x2-mi)/mami
           x2=x2.reshape((len(x2),1))
       else:       
           for i in range(len(x2)):
                if (max(x2[i])-min(x2[i]))==0:
                    x2[i]=(x2[i]-x2[i])
                else:
                    x2[i] =(x2[i]-min(x2[i]))/(max(x2[i])-min(x2[i]))
       x =np.hstack((x1,x2))
    else:
       x= x1
    dataset = LoadDataset(x)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_old=0
    for epoch in range(40):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).float()
            x_bar, z = model(x)
            loss_new = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss_new))
        if epoch >20:
            if loss_new<loss_old:
                z_new=z
                torch.save(model.state_dict(), output_path+prefix+".pkl")
        loss_old =loss_new
    return z_new







