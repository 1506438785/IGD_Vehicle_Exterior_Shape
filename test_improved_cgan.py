import torch
import numpy as np
import os
import torch.nn as nn

# 定义生成器和判别器模型类（需要与训练中的定义一致）
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

    def forward(self, noise, labels):
        labels_flat = labels.view(labels.size(0), -1)
        gen_input = torch.cat((noise, labels_flat), -1)
        point_cloud = self.model(gen_input)
        point_cloud = point_cloud.view(point_cloud.size(0), 2048, 3)
        return point_cloud

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

    def forward(self, point_cloud, labels):
        point_cloud_flat = point_cloud.view(point_cloud.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        d_in = torch.cat((point_cloud_flat, labels_flat), -1)
        validity = self.model(d_in)
        return validity

# 超参数
latent_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练模型
generator = Generator(latent_dim=latent_dim).to(device)
discriminator = Discriminator(input_dim=2048*3*2).to(device)

generator.load_state_dict(torch.load('checkpoints/gbest_model_5000.pth'))
discriminator.load_state_dict(torch.load('checkpoints/dbest_model_5000.pth'))

generator.eval()
discriminator.eval()
# 把文件名作为一个变量
filename = "aj00104.npy"

# 假设你的文件路径是 pre_data/data_label/label
file_path = "models_labels_npy"

# 使用 os.path.join 来拼接路径
full_path = os.path.join(file_path, filename)

# 然后再读取文件
test_labels = np.load(full_path)
# 生成测试数据
# test_labels = np.load('pre_data/data_label/label/cs00100.npy')
# test_labels_tensor = torch.tensor(test_labels).float().to(device)
test_labels_tensor = torch.tensor(test_labels).float().unsqueeze(0).to(device)  # 添加unsqueeze(0)
# 生成噪声
batch_size = test_labels_tensor.size(0)
z = torch.randn(batch_size, latent_dim).to(device)

# 生成点云
with torch.no_grad():
    gen_point_clouds = generator(z, test_labels_tensor)
gen_point_clouds = gen_point_clouds.cpu().numpy()

num_samples = 1  # 生成10个点云数据


# 保存生成的点云数据
output_dir = 'generated_point_clouds'
os.makedirs(output_dir, exist_ok=True)

for i, point_cloud in enumerate(gen_point_clouds):
    file_path = os.path.join(output_dir, f'{filename}-point_cloud_{i}.npy')
    np.save(file_path, point_cloud)
    print(f'Saved {file_path}')
#
