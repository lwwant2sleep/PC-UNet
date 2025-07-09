import torch

class ChannelSELayer(torch.nn.Module):

    def __init__(self, num_channels):

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out



class HANCLayer(torch.nn.Module):

    def __init__(self, in_chnl, out_chnl, k):

        super(HANCLayer, self).__init__()

        self.k = k

        self.cnv = torch.nn.Conv2d((2 * k - 1) * in_chnl, out_chnl, kernel_size=(1, 1))
        self.act = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm2d(out_chnl)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        x = inp

        if self.k == 1:
            x = inp

        elif self.k == 2:
            x = torch.cat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                ],
                dim=2,
            )

        elif self.k == 3:
            x = torch.cat(
                [
                    x,
                    torch.nn.Upsample(scale_factor=2)(torch.nn.AvgPool2d(2)(x)),
                    torch.nn.AvgPool2d(2)(torch.nn.Upsample(scale_factor=2)(x)),#x

                    torch.nn.Upsample(scale_factor=2)(torch.nn.MaxPool2d(2)(x)),
                    torch.nn.MaxPool2d(2)(torch.nn.Upsample(scale_factor=2)(x)),#x

                ],
                dim=2,
            )



        x = x.view(batch_size, num_channels * (2 * self.k - 1), H, W)

        x = self.act(self.bn(self.cnv(x)))

        return x



class Conv2d_batchnorm(torch.nn.Module):


    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
    ):

        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.sqe(self.activation(x))


class Conv2d_channel(torch.nn.Module):


    def __init__(self, num_in_filters, num_out_filters):

        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=(1, 1),
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)
        self.sqe = ChannelSELayer(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)
        return self.sqe(self.activation(x))



class Chanel_shuffle_ave(torch.nn.Module):
    def __init__(self,n_filts):
        super().__init__()
        self.n_filts=n_filts
    def forward(self,inp):
        batchsize, num_channels, height, width = inp.size()
        x=inp

        channels_per_group = num_channels // self.n_filts
        x = x.view(batchsize, self.n_filts, channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(batchsize, -1, height, width)

        x = x.view(batchsize, self.n_filts, channels_per_group, height, width)

        x = torch.mean(x, dim=2)
        x = x.view(batchsize, -1, height, width)

        return x


class HANCBlock(torch.nn.Module):


    def __init__(self, n_filts, out_channels, k=3, inv_fctr=3):


        super().__init__()

        self.conv1 = torch.nn.Conv2d(n_filts, n_filts * inv_fctr, kernel_size=1)
        self.norm1 = torch.nn.BatchNorm2d(n_filts * inv_fctr)

        self.channel_sa=Chanel_shuffle_ave(n_filts)

        self.conv2 = torch.nn.Conv2d(
            n_filts,
            n_filts,
            kernel_size=3,
            padding=1,
            groups=n_filts,
        )

        self.norm2 = torch.nn.BatchNorm2d(n_filts)

        self.hnc = HANCLayer(n_filts, n_filts, k)

        self.norm = torch.nn.BatchNorm2d(n_filts)

        self.conv3 = torch.nn.Conv2d(n_filts, out_channels, kernel_size=1)
        self.norm3 = torch.nn.BatchNorm2d(out_channels)

        self.sqe = ChannelSELayer(out_channels)

        self.activation = torch.nn.LeakyReLU()


    def forward(self, inp):

        x = self.conv1(inp)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.channel_sa(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.hnc(x)

        x = self.norm(x + inp)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.activation(x)
        x = self.sqe(x)

        return x



class ResPath(torch.nn.Module):


    def __init__(self, in_chnls, n_lvl):


        super(ResPath, self).__init__()

        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])
        self.sqes = torch.nn.ModuleList([])

        self.bn = torch.nn.BatchNorm2d(in_chnls)
        self.act = torch.nn.LeakyReLU()
        self.sqe = torch.nn.BatchNorm2d(in_chnls)

        for i in range(n_lvl):
            self.convs.append(
                torch.nn.Conv2d(in_chnls, in_chnls, kernel_size=(3, 3), padding=1)
            )
            self.bns.append(torch.nn.BatchNorm2d(in_chnls))
            self.sqes.append(ChannelSELayer(in_chnls))


    def forward(self, x):

        for i in range(len(self.convs)):
            x = x + self.sqes[i](self.act(self.bns[i](self.convs[i](x))))

        return self.sqe(self.act(self.bn(x)))




class PC_UNet(torch.nn.Module):


    def __init__(self, n_channels, n_classes, n_filts=32):


        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.pool = torch.nn.MaxPool2d(2)

        self.cnv11 = HANCBlock(n_channels, n_filts, k=3, inv_fctr=3)
        self.cnv12 = HANCBlock(n_filts, n_filts, k=3, inv_fctr=3)

        self.cnv21 = HANCBlock(n_filts, n_filts * 2, k=3, inv_fctr=3)
        self.cnv22 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3)

        self.cnv31 = HANCBlock(n_filts * 2, n_filts * 4, k=3, inv_fctr=3)
        self.cnv32 = HANCBlock(n_filts * 4, n_filts * 4, k=3, inv_fctr=3)

        self.cnv41 = HANCBlock(n_filts * 4, n_filts * 8, k=3, inv_fctr=3)
        self.cnv42 = HANCBlock(n_filts * 8, n_filts * 8, k=3, inv_fctr=3)

        self.cnv51 = HANCBlock(n_filts * 8, n_filts * 16, k=3, inv_fctr=3)
        self.cnv52 = HANCBlock(n_filts * 16, n_filts * 16, k=3, inv_fctr=3)

        self.rspth1 = ResPath(n_filts, 4)
        self.rspth2 = ResPath(n_filts * 2, 3)
        self.rspth3 = ResPath(n_filts * 4, 2)
        self.rspth4 = ResPath(n_filts * 8, 1)


        self.up6 = torch.nn.ConvTranspose2d(n_filts * 16, n_filts * 8, kernel_size=(2, 2), stride=2)
        self.cnv61 = HANCBlock(n_filts * 8 + n_filts * 8, n_filts * 8, k=3, inv_fctr=3)
        self.cnv62 = HANCBlock(n_filts * 8, n_filts * 8, k=3, inv_fctr=3)

        self.up7 = torch.nn.ConvTranspose2d(n_filts * 8, n_filts * 4, kernel_size=(2, 2), stride=2)
        self.cnv71 = HANCBlock(n_filts * 4 + n_filts * 4, n_filts * 4, k=3, inv_fctr=3)
        self.cnv72 = HANCBlock(n_filts * 4, n_filts * 4, k=3, inv_fctr=34)

        self.up8 = torch.nn.ConvTranspose2d(n_filts * 4, n_filts * 2, kernel_size=(2, 2), stride=2)
        self.cnv81 = HANCBlock(n_filts * 2 + n_filts * 2, n_filts * 2, k=3, inv_fctr=3)
        self.cnv82 = HANCBlock(n_filts * 2, n_filts * 2, k=3, inv_fctr=3)

        self.up9 = torch.nn.ConvTranspose2d(n_filts * 2, n_filts, kernel_size=(2, 2), stride=2)
        self.cnv91 = HANCBlock(n_filts + n_filts, n_filts, k=3, inv_fctr=3)
        self.cnv92 = HANCBlock(n_filts, n_filts, k=3, inv_fctr=3)

        if n_classes == 1:
            self.out = torch.nn.Conv2d(n_filts, n_classes, kernel_size=(1, 1))
            self.last_activation = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Conv2d(n_filts, n_classes + 1, kernel_size=(1, 1))
            self.last_activation = None

        self.id = torch.nn.Identity()

    def forward(self, x):

        x1 = x


        x2 = self.cnv11(x1)
        x2 = self.cnv12(x2)

        x2p = self.pool(x2)

        x3 = self.cnv21(x2p)
        x3 = self.cnv22(x3)

        x3p = self.pool(x3)

        x4 = self.cnv31(x3p)
        x4 = self.cnv32(x4)

        x4p = self.pool(x4)

        x5 = self.cnv41(x4p)
        x5 = self.cnv42(x5)

        x5p = self.pool(x5)

        x6 = self.cnv51(x5p)
        x6 = self.cnv52(x6)

        x2 = self.rspth1(x2)
        x3 = self.rspth2(x3)
        x4 = self.rspth3(x4)
        x5 = self.rspth4(x5)


        x7 = self.up6(x6)
        x7 = self.cnv61(torch.cat([x7, x5], dim=1))
        x7 = self.cnv62(x7)

        x8 = self.up7(x7)
        x8 = self.cnv71(torch.cat([x8, x4], dim=1))
        x8 = self.cnv72(x8)

        x9 = self.up8(x8)
        x9 = self.cnv81(torch.cat([x9, x3], dim=1))
        x9 = self.cnv82(x9)

        x10 = self.up9(x9)
        x10 = self.cnv91(torch.cat([x10, x2], dim=1))
        x10 = self.cnv92(x10)

        if self.last_activation is not None:
            logits = self.last_activation(self.out(x10))

        else:
            logits = self.out(x10)

        logits = self.id(logits)

        return logits
