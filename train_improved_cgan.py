import logging
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.optim as optim
import torch.nn as nn
from scipy.spatial import distance_matrix
from scipy.stats import wasserstein_distance
from datetime import datetime

class PointCloudDataset(Dataset):
    def __init__(self, file_paths, label_paths):
        self.file_paths = file_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        data_tensor = torch.tensor(data).float()

        label = np.load(self.label_paths[idx])
        label_tensor = torch.tensor(label).float()

        return data_tensor, label_tensor

def get_npy_files(folder_path):
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
    return file_paths

# folder_path = 'pre_data/data_npy/chengshu'
folder_path = 'pre_data/data_npy/anjing'

label_path = 'pre_data/data_label/label'
file_paths = get_npy_files(folder_path)
label_paths = get_npy_files(label_path)

dataset = PointCloudDataset(file_paths, label_paths)
data_loader = DataLoader(dataset, batch_size=27, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self, input_dim=2048 * 3 * 2):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.apply(self.weights_init)

    def forward(self, point_cloud, labels):
        print(point_cloud.shape, labels.shape)
        point_cloud_flat = point_cloud.view(point_cloud.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        d_in = torch.cat((point_cloud_flat, labels_flat), -1)
        validity = self.model(d_in)
        return validity

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=2048 * 3):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + 2048 * 3, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        self.apply(self.weights_init)

    def forward(self, noise, labels):
        labels_flat = labels.view(labels.size(0), -1)
        gen_input = torch.cat((noise, labels_flat), -1)
        point_cloud = self.model(gen_input)
        point_cloud = point_cloud.view(point_cloud.size(0), 2048, 3)
        return point_cloud

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

def chamfer_distance(pc1, pc2):
    dist1 = torch.cdist(pc1, pc2, p=2)
    dist2 = torch.cdist(pc2, pc1, p=2)
    chamfer_dist = (dist1.min(dim=2)[0].mean() + dist2.min(dim=2)[0].mean()) / 2
    return chamfer_dist * 100

def emd_loss(real_point_cloud, gen_point_cloud):
    emd = 0
    for i in range(real_point_cloud.size(0)):
        emd += wasserstein_distance(real_point_cloud[i].cpu().detach().numpy().flatten(),
                                    gen_point_cloud[i].cpu().detach().numpy().flatten())
    emd = torch.tensor(emd / real_point_cloud.size(0), requires_grad=True).to(real_point_cloud.device)
    return emd * 100

def combined_loss(real_point_cloud, gen_point_cloud, alpha=0.5):
    cd = chamfer_distance(real_point_cloud, gen_point_cloud)
    emd = emd_loss(real_point_cloud, gen_point_cloud)
    return alpha * cd + (1 - alpha) * emd

# 超参数
lr = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 100
epochs = 2000000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator(input_dim=2048*3*2).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# 获取当前日期和时间
current_time = datetime.now().strftime('%Y%m%d_%H%M')

# 创建新的文件夹
checkpoint_dir = f'checkpoints_{current_time}'
os.makedirs(checkpoint_dir, exist_ok=True)

min_loss = float('inf')
dbest_save_path = os.path.join(checkpoint_dir, 'discriminator_best.pth')
gbest_save_path = os.path.join(checkpoint_dir, 'generator_best.pth')
dlast_save_path = os.path.join(checkpoint_dir, 'discriminator_last.pth')
glast_save_path = os.path.join(checkpoint_dir, 'generator_last.pth')

log_file_path = os.path.join(checkpoint_dir, 'training_log.log')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file_path, filemode='w')
logger = logging.getLogger()

for epoch in range(epochs):
    for i, (real_point_clouds, labels) in enumerate(data_loader):
        batch_size = real_point_clouds.size(0)
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        real_point_clouds = real_point_clouds.to(device)
        labels = labels.to(device)

        # 训练生成器
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_point_clouds = generator(z, labels)
        g_loss = combined_loss(real_point_clouds, gen_point_clouds)

        g_loss.backward()
        optimizer_G.step()

        # 训练判别器
        optimizer_D.zero_grad()
        real_loss = combined_loss(real_point_clouds, real_point_clouds)
        fake_loss = combined_loss(real_point_clouds, gen_point_clouds.detach())
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    if epoch % 500 == 0:
        dbest_save_path = os.path.join(checkpoint_dir, f'dbest_model_{epoch}.pth')
        gbest_save_path = os.path.join(checkpoint_dir, f'gbest_model_{epoch}.pth')
        torch.save(discriminator.state_dict(), dbest_save_path)
        torch.save(generator.state_dict(), gbest_save_path)
    if epoch % 10 == 0:
        log_message = f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(data_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
        print(log_message)
        logger.info(log_message)
