import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)      # torch.Size([1, 1, 3, 3])
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)      # torch.Size([1, 1, 3, 3])
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)     # weight就是filter
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)      # torch.Size([8, 1, 480, 640])
class CannyFilter(nn.Module):
    def __init__(self,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda:3'
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        # gaussian

        gaussian_2D = self.get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,
                                         out_channels=1,
                                         kernel_size=k_gaussian,
                                         padding=k_gaussian // 2,
                                         bias=False)
        #with torch.no_grad():
            # self.gaussian_filter.weight[:] = torch.from_numpy(gaussian_2D)
        self.gaussian_filter.weight.data = torch.from_numpy(gaussian_2D.astype(np.float32)).reshape(-1, 1, 3, 3).cuda()
        # sobel

        sobel_2D = self.get_sobel_kernel(k_sobel)
        self.sobel_filter_x = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)

        #self.sobel_filter_x.weight[:] = torch.from_numpy(sobel_2D)
        self.sobel_filter_x.weight.data = torch.from_numpy(sobel_2D.astype(np.float32)).reshape(-1, 1, 3, 3).cuda()

        self.sobel_filter_y = nn.Conv2d(in_channels=1,
                                        out_channels=1,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        bias=False)

        #self.sobel_filter_y.weight[:] = torch.from_numpy(sobel_2D.T)
        self.sobel_filter_y.weight.data = torch.from_numpy(sobel_2D.T.astype(np.float32)).reshape(-1, 1, 3, 3).cuda()

        # thin

        thin_kernels = self.get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,
                                            out_channels=8,
                                            kernel_size=thin_kernels[0].shape,
                                            padding=thin_kernels[0].shape[-1] // 2,
                                            bias=False)

        #self.directional_filter.weight[:, 0] = torch.from_numpy(directional_kernels)
        self.directional_filter.weight.data = torch.from_numpy(directional_kernels.astype(np.float32)).reshape(-1, 1, 3, 3).cuda()
        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=3,
                                    padding=1,
                                    bias=False)

        #self.hysteresis.weight[:] = torch.from_numpy(hysteresis)
        self.hysteresis.weight.data = torch.from_numpy(hysteresis.astype(np.float32)).reshape(-1, 1, 3, 3).cuda()


    def forward(self, img, low_threshold=0.15, high_threshold=0.30, hysteresis=True):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).cuda()
        grad_x = torch.zeros((B, 1, H, W)).cuda()
        grad_y = torch.zeros((B, 1, H, W)).cuda()
        grad_magnitude = torch.zeros((B, 1, H, W)).cuda()
        grad_orientation = torch.zeros((B, 1, H, W)).cuda()

        # 1.对输入图像进行高斯平滑，降低错误率
        # gaussian blur
        for c in range(C):
            # 高斯滤波，获得blurred picture
            blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1].cuda()).cuda()
            # sobel滤波，提取梯度，目的是检测边缘
            grad_x = grad_x + self.sobel_filter_x(blurred[:, c:c + 1])
            grad_y = grad_y + self.sobel_filter_y(blurred[:, c:c + 1])

        # 2.计算梯度幅度和方向来估计每一点处的边缘强度与方向
        # thick edges 厚边
        grad_x, grad_y = grad_x / C, grad_y / C
        # magnitude 梯度的大小
        grad_magnitude = (grad_x ** 2 + grad_y ** 2) ** 0.5
        # orientation 梯度的方向
        grad_orientation = torch.atan(grad_y / grad_x)
        grad_orientation = grad_orientation * (360 / np.pi) + 180  # convert to degree
        grad_orientation = torch.round(grad_orientation / 45) * 45  # keep a split by 45

        # thin edges 薄边
        directional = self.directional_filter(grad_magnitude)
        # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()

        # 3.根据梯度方向，对梯度幅值进行非极大值抑制。本质上是对Sobel、Prewitt等算子结果的进一步细化
        # non maximum suppression direction by direction ，为了细化边缘，使用非极大值抑制方法
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non-maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        # 4.用双阈值处理和连接边缘
        # thresholds 阈值
        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                # 滞后
                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return blurred, grad_x, grad_y, grad_magnitude, grad_orientation, thin_edges

    def get_gaussian_kernel(self,k=3, mu=0, sigma=1, normalize=True):
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5

        # compute the 2 dimension gaussian

        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        return gaussian_2D

    def get_sobel_kernel(self,k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator
        return sobel_2D

    def get_thin_kernels(self,start=0, end=360, step=45):
        k_thin = 3  # actual size of the directional kernel
        # increase for a while to avoid interpolation when rotating
        k_increased = k_thin + 2

        # get 0° angle directional kernel
        thin_kernel_0 = np.zeros((k_increased, k_increased))
        thin_kernel_0[k_increased // 2, k_increased // 2] = 1
        thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

        # rotate the 0° angle directional kernel to get the other ones
        thin_kernels = []
        for angle in range(start, end, step):
            (h, w) = thin_kernel_0.shape
            # get the center to not rotate around the (0, 0) coord point
            center = (w // 2, h // 2)
            # apply rotation
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            kernel_angle_increased = cv2.warpAffine(thin_kernel_0, rotation_matrix, (w, h), cv2.INTER_NEAREST)

            # get the k=3 kernel
            kernel_angle = kernel_angle_increased[1:-1, 1:-1]
            is_diag = (abs(kernel_angle) == 1)  # because of the interpolation
            kernel_angle = kernel_angle * is_diag  # because of the interpolation
            thin_kernels.append(kernel_angle)
        return thin_kernels


# 融合ECA的局部，全局注意力模块
class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        """
        ECA 模块初始化。
        Args:
            channel (int): 输入特征图的通道数。
            gamma (int): 控制动态核大小。
            b (int): 核大小调整的偏移量。
        """
        super(ECALayer, self).__init__()
        # 动态调整核大小 k
        k = int(abs((torch.log(torch.tensor(channel, dtype=torch.float32)) / torch.log(torch.tensor(2.0))) + b) // gamma + 1)
        if k % 2 == 0:
            k += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)  # 1D 卷积

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)  # 转换为 (B, 1, C)
        y = self.conv(y)  # 1D 卷积提取通道关系
        y = y.transpose(-1, -2).unsqueeze(-1)  # 恢复到 (B, C, 1, 1)
        y = torch.sigmoid(y)  # Sigmoid 激活
        return x * y  # 按通道权重加权


class SpectralBlock(nn.Module):
    def __init__(self, num_channels, reduction=16, patch_size=32):
        super(SpectralBlock, self).__init__()
        self.patch_size = patch_size

        # 替换局部多头注意力为 ECA 注意力
        self.eca_attention = ECALayer(num_channels)

        # 全局注意力模块（通道注意力）
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels // reduction, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(num_channels // reduction, num_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # self.global_attention_adjust = nn.Conv2d(num_channels, num_channels * 2, kernel_size=1)  # 调整通道数

        # 动态卷积模块
        self.dynamic_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * 2, kernel_size=3, padding=1, groups=num_channels),
            nn.Mish(),
            nn.Conv2d(num_channels * 2, num_channels, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        b, c, h, w = x.size()

        # ===== 替换局部多头注意力为 ECA 注意力 =====
        local_attention = self.eca_attention(x)

        # ===== 全局注意力 =====
        global_attention = self.global_attention(x)  # (b, c, 1, 1)
        # global_attention = self.global_attention_adjust(global_attention)  # 调整通道数
        enhanced_global = local_attention * global_attention

        # ===== 动态卷积 =====
        dynamic_out = self.dynamic_conv(enhanced_global)

        return dynamic_out






class UpSample(nn.Module):
    def __init__(self,scale_ratio):
        super(UpSample,self).__init__()
        self.scale_ratio = scale_ratio

    def forward(self,x):
        x = F.interpolate(x, scale_factor=self.scale_ratio, mode='bicubic')
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.spatial_block = SpatialBlock(in_channels, out_channels)
        self.spectral_block = SpectralBlock(in_channels)

    def forward(self, x):
        spatial = self.spatial_block(x)
        spectral = self.spectral_block(spatial)
        return spatial


class DeepResidualNetwork(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels):
        super(DeepResidualNetwork, self).__init__()
        self.blocks = nn.Sequential(
            *[ResidualBlock(in_channels, out_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)



class MyModel(nn.Module):
    def __init__(self, scale_ratio, n_select_bands, n_bands):
        super(MyModel, self).__init__()
        self.n_bands = n_bands
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands

        # MSI 分支
        self.msi_conv2d = nn.Conv2d(self.n_select_bands, n_bands, kernel_size=3, padding=1, stride=1)
        self.msi_branch = nn.Sequential(
            SpatialBlock(in_channels=n_bands, out_channels=n_bands)
        )

        # HSI 分支
        self.hsi_conv2d_upsample = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1, stride=1),
            UpSample(self.scale_ratio)
        )
        self.hsi_branch = nn.Sequential(
            SpectralBlock(n_bands)
        )

        # 插值分支
        self.interpolate_conv2d = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, padding=1, stride=1)
        )
        self.interpolate_branch = nn.Sequential(
            SpatialBlock(in_channels=n_bands, out_channels=n_bands),
            SpectralBlock(n_bands)
        )

        # 主干网络前的融合
        self.pre_fusion = nn.Conv2d(n_bands * 3, n_bands, kernel_size=1, padding=0, stride=1)

        # 主干网络
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=1, padding=0, stride=1),
            nn.Mish()
        )

        # 深度残差网络
        self.deep_residual_network = DeepResidualNetwork(
            num_blocks=1,  # 残差块的数量
            in_channels=n_bands,
            out_channels=n_bands
        )

        # 动态融合和输出
        self.fusion_conv = nn.Conv2d(n_bands * 2, n_bands, kernel_size=1, padding=0)
        self.conv2d_3 = nn.Conv2d(n_bands, n_bands, kernel_size=1, padding=0, stride=1)

    def spatial_canny(self, x):
        canny = CannyFilter()
        _, _, _, _, _, thin_edges = canny(x)
        return thin_edges

    def forward(self, x_lr_hsi, x_hr_msi, x_hssi):
        # MSI 分支
        out_msi = self.msi_conv2d(x_hr_msi)
        out_msi = out_msi + self.msi_branch(out_msi)

        # HSI 分支
        out_hsi = self.hsi_conv2d_upsample(x_lr_hsi)
        out_hsi = out_hsi + self.hsi_branch(out_hsi)

        # 插值分支
        out_mix = self.interpolate_conv2d(x_hssi)
        out_mix = out_mix + self.interpolate_branch(out_mix)

        # 主干网络前融合
        combined = torch.cat([out_msi, out_hsi, out_mix], dim=1)
        fused_features = self.pre_fusion(combined)

        # 主干网络
        in_global = self.conv2d_1(fused_features)

        # 深度残差网络
        out_spatial = self.deep_residual_network(in_global)
        out_spatial_for_loss = out_spatial  # 用于计算空间损失

        # 动态融合
        fusion_input = torch.cat([in_global, out_spatial], dim=1)
        fusion_weight = torch.sigmoid(self.fusion_conv(fusion_input))
        out_spectral = fusion_weight * in_global + (1 - fusion_weight) * out_spatial
        out_spectral_for_loss = out_spectral  # 用于计算光谱损失

        # 最终输出
        out = self.conv2d_3(out_spectral)

        # 空间特征检测（Sobel 或 Canny）
        spat_canny = self.spatial_canny(out_spatial_for_loss)

        return out, spat_canny, out_spectral_for_loss





class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.mish = nn.Mish()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.global_pool(x)
        scale = self.fc1(scale)
        scale = self.mish(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale



# 2022年论文ParC+自己设计的多尺度qkv
class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, reduction=16):
        super(SpatialBlock, self).__init__()

        # 水平循环卷积
        self.horizontal_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2), groups=in_channels
        )

        # 垂直循环卷积
        self.vertical_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0), groups=in_channels
        )

        # 点卷积
        self.pointwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 全局注意力（使用 SE 模块）
        self.global_attention = MultiscaleBlock(out_channels)

        # 动态卷积
        self.dynamic_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Mish(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # BatchNorm 和激活函数
        self.norm = nn.BatchNorm2d(out_channels)
        self.mish = nn.Mish()

    def forward(self, x):
        residual = x

        # 水平和垂直循环卷积
        horizontal_out = self.horizontal_conv(x)
        vertical_out = self.vertical_conv(horizontal_out)

        # 点卷积
        out = self.pointwise_conv(vertical_out)

        # 全局注意力（QKV增强注意力）
        out = self.global_attention(out)

        # 动态卷积
        out = self.dynamic_conv(out)

        # BatchNorm 和激活
        out = self.norm(out)
        out = self.mish(out)

        # 残差连接
        return out + residual


class MultiscaleBlock(nn.Module):
    def __init__(self, in_channels):
        super(MultiscaleBlock, self).__init__()
        self.in_channels = in_channels

        # 三个不同尺度的卷积核
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels)

        # 池化操作，使得 q2, k2, v2 的形状为 (B, 1, H, W)
        self.pool_q = nn.AdaptiveAvgPool2d((None, None))
        self.pool_k = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_v = nn.AdaptiveAvgPool2d((None, None))

        # 用于调整注意力权重的线性层
        self.query_proj = nn.Conv2d(1, 1, kernel_size=1)
        self.key_proj = nn.Conv2d(1, 1, kernel_size=1)
        self.value_proj = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        # 多尺度卷积
        q1 = self.conv3x3(x)  # (B, C, H, W)
        k1 = self.conv5x5(x)
        v1 = self.conv7x7(x)

        # 多尺度池化
        q2 = self.pool_q(q1).mean(dim=1, keepdim=True)  # (B, 1, H, W)
        k2 = self.pool_k(k1).mean(dim=1, keepdim=True)
        v2 = self.pool_v(v1).mean(dim=1, keepdim=True)  # (B, 1, 1, 1)


        # 注意力机制
        q2_proj = self.query_proj(q2).flatten(2)  # (B, 1, H*W)
        k2_proj = self.key_proj(k2).flatten(2)  # (B, 1, H*W)
        v2_proj = self.value_proj(v2).flatten(2)  # (B, 1, H*W)

        # 计算注意力权重
        attn = torch.matmul(q2_proj, k2_proj.transpose(-2, -1)) / (H * W) ** 0.5
        attn = F.softmax(attn, dim=-1)  # (B, 1, 1)
        atte = torch.matmul(attn, v2_proj).view(B, 1, H, W)  # (B, 1, H, W)

        # 注意力施加到 x 上
        x_after_atte = x * atte.expand(-1, C, -1, -1)  # 广播 atte 到所有通道

        return x_after_atte


def get_model_flops(model, input_tensor1,input_tensor2):
    # 将模型和输入张量移动到CPU
    model.eval()
    model.apply(lambda m: m.train(False))
    input_tensor1 = input_tensor1.detach()
    input_tensor2 = input_tensor2.detach()

    # 计算FLOPs
    flops, params = profile(model.to('cpu'), (input_tensor1,input_tensor2), verbose=False)

    # 将FLOPs转换为十亿
    flops = flops / 1e9 * 2  # 乘以2是因为乘法和加法
    return flops, params

if __name__ == '__main__':

    # lr_hsi = torch.rand([1,102,273,178])
    # lr_hsi_1 = torch.tensor([1,102,273,178])
    # # lr = torch.rand([1,102,100,512])
    # hr_msi = torch.rand([1,5,1092,712])
    #
    #
    # x = torch.randn([1, 102, 1092, 712])
    #
    # model = MyModel(scale_ratio=4,n_select_bands=3,n_bands=103)
    #
    #
    # time1 =time.time()
    # output_tensor,_,_ = model(lr_hsi,hr_msi)
    # time2 = time.time() - time1
    # print("time : ",time2*1000)
    # print(output_tensor.shape)  # 输出张量的形状应该是[1, 32, 32, 32]
    #
    # flops, params = get_model_flops(model, lr_hsi,hr_msi)
    # print(f'Total GFLOPS: {flops:.5f}')
    # print(f'Total number of parameters: {params / 1e6} M')

    x = torch.randn(1,103,128,128)
    block = MultiscaleBlock(103)
    out = block(x)
    print(out.shape)
