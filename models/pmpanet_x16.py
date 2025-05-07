import torch
import torch.nn as nn

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DilaConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, dilation=2, bias=True, activation='prelu', norm=None):
        super(DilaConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding,  dilation, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class channel_attentionBlock(torch.nn.Module):
    def __init__(self, num_filter):
        super(channel_attentionBlock, self).__init__()

        self.g_aver_pooling1 = torch.nn.AdaptiveAvgPool2d(1)

        self.fc1 = torch.nn.Linear(in_features=num_filter, out_features=round(num_filter / 16))

        self.act_1 = torch.nn.ReLU(True)

        self.fc2 = torch.nn.Linear(in_features=round(num_filter / 16), out_features=num_filter)

        self.act_2 = torch.nn.Sigmoid()

        # self.avgpool_1 = torch.nn.AvgPool2d(8, 4, 2)

        # self.up_1 = DeconvBlock(num_filter, num_filter , kernel_size, stride, padding, activation='prelu', norm=None)

        # self.act_1 = torch.nn.ReLU(True)

    def forward(self, x):

        x1 = self.g_aver_pooling1(x)
        x1 = x1.view(x1.size(0), -1)
        c1 = self.fc1(x1)
        act1 = self.act_1(c1)
        c2 = self.fc2(act1)
        act2 = self.act_2(c2)
        act2 = act2.view(act2.size(0), act2.size(1), 1, 1)

        y = x + x*act2

        return y

class MultiViewBlock1(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock1, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(2 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(3 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(4 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(5 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        # print(concat_p1.shape, '111111111111111111')
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class MultiViewBlock2(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock2, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(3 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(4 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(5 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(6 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)
        # out = self.output_conv1(h_prior1)

        return h_prior1


class MultiViewBlock3(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock3, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(4 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(5 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(6 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(7 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1


class FeedbackBlock1(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(FeedbackBlock1, self).__init__()
        self.conv1 = ConvBlock(in_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1 * (act1 * x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0


class FeedbackBlock2(torch.nn.Module):
    def __init__(self, in_filter, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu',
                 norm=None):
        super(FeedbackBlock2, self).__init__()
        self.down1 = ConvBlock(in_filter, num_filter, kernel_size, stride, padding, activation='prelu', norm=None)
        self.conv1 = ConvBlock(num_filter, num_filter, 1, 1, 0, activation='prelu', norm=None)
        self.avgpool_1 = torch.nn.AvgPool2d(4, 4, 0)
        self.up_1 = DeconvBlock(num_filter, num_filter, 8, 4, 2, activation='prelu', norm=None)
        self.act_1 = torch.nn.ReLU(True)

        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.down1(x)
        x = self.conv1(x)
        p1 = self.avgpool_1(x)
        l00 = self.up_1(p1)
        act1 = self.act_1(x - l00)
        out_la = x + 0.1 * (act1 * x)

        h0 = self.up_conv1(out_la)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - out_la)
        return h1 + h0

class MultiViewBlock4(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=12, stride=8, padding=2, bias=True, activation='prelu', norm=None):
        super(MultiViewBlock4, self).__init__()

        self.dilaconv1 = DilaConvBlock(num_filter, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.dilaconv2 = DilaConvBlock(5 * 64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        self.dilaconv3 = DilaConvBlock(6 * 64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        self.dilaconv4 = DilaConvBlock(7 * 64, 64, 3, 1, 4, dilation=4, activation='prelu', norm=None)
        self.dilaconv1_2 = DilaConvBlock(8 * 64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)
        self.direct_up1 = DeconvBlock(64, 64, kernel_size, stride, padding, activation='prelu', norm=None)
        # self.dilaconv5 = DilaConvBlock(64, 64, 3, 1, 3, dilation=3, activation='prelu', norm=None)
        # # self.dilaconv6 = DilaConvBlock(64, 64, 3, 1, 2, dilation=2, activation='prelu', norm=None)
        # # self.dilaconv7 = DilaConvBlock(64, 64, 3, 1, 1, dilation=1, activation='prelu', norm=None)

        # self.output_conv1 = ConvBlock(num_filter, out_num_filter, 3, 1, 1, activation='prelu', norm=None)

    def forward(self, x):
        x_prior1 = self.dilaconv1(x)
        concat1 = torch.cat((x, x_prior1), 1)
        x_prior2 = self.dilaconv2(concat1)
        concat2 = torch.cat((concat1, x_prior2), 1)
        x_prior3 = self.dilaconv3(concat2)
        concat3 = torch.cat((concat2, x_prior3), 1)
        x_prior4 = self.dilaconv4(concat3)
        concat_p1 = torch.cat((concat3, x_prior4), 1)
        x_prior1_2 = self.dilaconv1_2(concat_p1)

        h_prior1 = self.direct_up1(x_prior1_2)

        return h_prior1

class Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(Net, self).__init__()
        
        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        #### 
        elif scale_factor == 16:
          kernel = 20
          stride = 16
          padding = 2
        
        #Initial Feature Extraction
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)

        self.feat_color0 = ConvBlock(3, feat, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color1 = ConvBlock(feat, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color2 = ConvBlock(base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.feat_color3 = ConvBlock(base_filter, base_filter, 1, 1, 0, activation='prelu', norm=None)

        #Multi-view prior
        self.m1 = MultiViewBlock1(64, 20, 16, 2)
        self.m2 = MultiViewBlock2(2*64, 20, 16, 2)
        self.m3 = MultiViewBlock3(3*64, 20, 16, 2)
        self.m4 = MultiViewBlock4(4*64, 20, 16, 2)
        #self.m5 = MultiViewBlock5(5*64, 8, 4, 2)

        #Channel_attention
        self.c1 = channel_attentionBlock(64)
        self.c2 = channel_attentionBlock(64)
        self.c3 = channel_attentionBlock(64)
        self.c4 = channel_attentionBlock(64)
        self.c5 = channel_attentionBlock(64)

        #Reconstruction 1
        self.r1_1 = FeedbackBlock1(64, base_filter, kernel, stride, padding)
        self.r1_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r1_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r1_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r1_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        #Reconstruction 2
        self.r2_1 = FeedbackBlock1(2*64, base_filter, kernel, stride, padding)
        self.r2_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r2_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r2_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r2_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        #Reconstruction 3
        self.r3_1 = FeedbackBlock1(3*64, base_filter, kernel, stride, padding)
        self.r3_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r3_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r3_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        self.r3_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        #Reconstruction 4
        # self.r4_1 = FeedbackBlock1(4*64, base_filter, kernel, stride, padding)
        # self.r4_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r4_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r4_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r4_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)

        #Reconstruction 5
        # self.r5_1 = FeedbackBlock1(5*64, base_filter, kernel, stride, padding)
        # self.r5_2 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r5_3 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r5_4 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)
        # self.r5_5 = FeedbackBlock2(base_filter, base_filter, kernel, stride, padding)


        self.down2 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.down3 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.down4 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        self.down4 = ConvBlock(base_filter, base_filter, kernel, stride, padding, activation='prelu', norm=None)
        #Reconstruction

        self.output_conv1_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv2_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv3_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv4_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv5_1 = ConvBlock(4*base_filter, base_filter, 3, 1, 1, activation='prelu', norm=None)
        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            
    def forward(self, rgb, depth):


        x = self.feat0(depth)
        x = self.feat1(x)

        c = self.feat_color0(rgb)
        c1 = self.feat_color1(c)
        c3 = self.feat_color3(c1)

 ############1
        mv1 = self.m1(x)
        rb1 = self.r1_1(x)
        rb2 = self.r1_2(rb1)
        rb3 = self.r1_3(rb2)
        rb4 = self.r1_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4),1)
        r1 = self.output_conv1_1(concat_h)
        
        d1 = mv1 + r1 + c3
        d1 = self.c1(d1)
##############2
        x2 = self.down2(d1)
        x2 = torch.cat((x, x2),1)
        mv2 = self.m2(x2)
        rb1 = self.r2_1(x2)
        rb2 = self.r2_2(rb1)
        rb3 = self.r2_3(rb2)
        rb4 = self.r2_4(rb3)
        concat_h = torch.cat((rb1, rb2, rb3, rb4),1)
        r2 = self.output_conv2_1(concat_h)

        d2 = mv2 + r2
        d2 = self.c2(d2)
##############3
        # x3 = self.down3(d2)
        # x3 = torch.cat((x2, x3),1)
        # #print(x3.shape, '3333333333333333')
        # mv3 = self.m3(x3)
        # rb1 = self.r3_1(x3)
        # rb2 = self.r3_2(rb1)
        # rb3 = self.r3_3(rb2)
        # rb4 = self.r3_4(rb3)
        # # rb5 = self.r3_5(rb4)
        # concat_h = torch.cat((rb1, rb2, rb3, rb4),1)
        # r3 = self.output_conv3_1(concat_h)

        # d3 = mv3 + r3
        # d3 = self.c3(d3)
        d = self.output_conv(d2)
        #print(d.shape,'dddddddddddddd33333333333333333')
        return d