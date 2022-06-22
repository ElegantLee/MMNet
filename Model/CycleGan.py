import torch.nn.functional as F
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

# distribution generator
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)
        return x


# metric generator
class ML512(nn.Module):
    def __init__(self, in_channal, out_dim):
        super(ML512, self).__init__()
        dim = 32

        self.image_to_features = nn.Sequential(
            nn.Conv2d(in_channal, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8 * dim, 16 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16 * dim, 32 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32 * dim, 64 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64 * dim, 128 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

        self.features_to_prob = nn.Linear(32 * 4 * dim, out_dim)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        x = self.features_to_prob(x)
        x = F.normalize(x)

        return x


class MLNet(nn.Module):
    def __init__(self, in_channal, out_dim):
        super(MLNet, self).__init__()
        dim = 32

        model = [nn.Conv2d(in_channal, dim, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(dim, dim * 2, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(dim * 2, dim * 4, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(dim * 4, dim * 4, 3, 1, 1),
                 nn.PReLU(),
                 nn.Conv2d(dim * 4, dim * 8, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 nn.Conv2d(dim * 8, dim * 8, 3, 1, 1),
                 nn.PReLU(),
                 nn.Conv2d(dim * 8, dim * 16, 3, 1, 1),
                 nn.PReLU(),
                 nn.MaxPool2d(kernel_size=2, stride=2),
                 ]

        self.image_to_features = nn.Sequential(*model)
        self.features_to_prob = nn.Sequential(
            nn.Linear(1024 * dim, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        x = self.features_to_prob(x)
        x = F.normalize(x)
        return x

class MLUnet(nn.Module):
    def __init__(self, in_channal, out_dim):
        ## con0_conv1_pool1
        super().__init__()
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channal, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(2))
        ##conv2_pool2
        self.encode2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(2))
        ##conv3_pool3
        self.encode3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(2))
        ##conv4_pool4
        self.encode4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(2))
        ##conv5_pool5
        self.encode5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(2))

        self.features_to_prob = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        x = self.encode1(x)
        x = self.encode2(x)
        x = self.encode3(x)
        x = self.encode4(x)
        x = self.encode5(x)
        x = x.view(x.size(0), -1)
        x = self.features_to_prob(x)
        x = F.normalize(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]



        # FCN classification layer
        self.model = nn.Sequential(*model)
        self.last = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, x):
        x = self.model(x)
        x = self.last(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
