from .common import *
class Generator(nn.Module):
    def __init__(self,  weight, latent_size=300, num_dimension=300, imp_num=1574, char_num=26, normalize=True):
        super(Generator, self).__init__()
        self.z_dim = latent_size
        self.char_num = char_num
        self.imp_num = imp_num
        self.weight = torch.tensor(weight)
        self.emb_layer = ImpEmbedding(weight, deepsets=False, normalize=normalize)
        self.num_dimension = num_dimension
        self.layer1 = nn.Sequential(
            nn.Linear(self.z_dim + char_num , 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer2 = nn.Sequential(
            nn.Linear(self.num_dimension, 1500),
            nn.BatchNorm1d(1500),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer3 = nn.Sequential(
            nn.Linear(3000 , 128 * 16 * 16),
            nn.BatchNorm1d(128 * 16 * 16),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            # チャネル数を128⇒64に変える。
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,inplace=True))

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                else:
                    continue
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                else:
                    continue
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                else:
                    continue
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
                else:
                    continue

    def forward(self, noise, char_class, y_imp, emb=True):
        y_1 = self.layer1(torch.cat([noise, char_class], dim=1))  # (100,1,1)⇒(300,1,1)
        # 印象情報のw2v
        if emb:
            attr = self.emb_layer(y_imp)
        else:
            attr = y_imp
        y_2 = self.layer2(attr)  # (300,1,1)⇒(1500,1,1)
        x = torch.cat([y_1, y_2], dim = 1)  # y_1 + y_2=(1800,1,1)
        x = self.layer3(x)  # (1800,1,1)⇒(512*8,1,1)
        x = x.view(-1, 128, 16, 16)  # (512,8,8)
        x = self.layer4(x)  # (512,8,8)⇒(256,16,16)
        x = self.layer5(x)  # (256,16,16)⇒(128,32,32)
        return x, attr

class Discriminator(nn.Module):
    def __init__(self, num_dimension=300, imp_num=1430, char_num=26):
        super(Discriminator, self).__init__()
        self.num_dimension = num_dimension
        self.imp_num = imp_num
        self.char_num = char_num
        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(1 + char_num, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2,inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            # nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True)
            )


        self.fc_TF = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128 * 16 * 16, 1024)),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

        self.fc_class =nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(128 * 16 * 16, 1024)),
            # nn.BatchNorm1d(1024),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.imp_num)
        )

        self.init_weights()


    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm1d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(1.0, 0.02)
                module.bias.data.zero_()

    def forward(self, img, char_class):
        char = char_class.view(char_class.size(0), char_class.size(1), 1, 1).expand(-1, -1, img.size(2), img.size(3))
        x = self.layer1(torch.cat([img, char], dim=1))
        x = self.layer2(x)
        x = x.view(-1, 128 * 16 * 16)
        x_TF = self.fc_TF(x)
        x_class = self.fc_class(x)
        return x_TF.squeeze(), x_class.squeeze()
