import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from einops import rearrange, repeat
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, layer_sizes, last_actv=False):
        super(MLP, self).__init__()

        layers = []
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if last_actv or i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x


# Adapted from https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/models.py
class VAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, conditional=False, condition_size=1024):
        super().__init__()
        if conditional:
            assert condition_size > 0

        assert isinstance(encoder_layer_sizes, list)
        assert isinstance(latent_size, int)
        assert isinstance(decoder_layer_sizes, list)

        self.latent_size = latent_size

        self.encoder = Encoder(encoder_layer_sizes, latent_size, conditional, condition_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, conditional, condition_size)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()
        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += condition_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, condition_size):
        super().__init__()
        self.MLP = nn.Sequential()
        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z, c=None):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


# Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)))
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    """
    Encoder of PointNet.

    Different from the implementation from the repo below,
        we add an option to transform the points (default: False).

    `forward` method:
        Input: x (batch_size, channel, num_points)
        Output: feature, transformation, transformed_feature

    Ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py
    """

    def __init__(self, global_feat=True, point_transform=False, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.point_transform = point_transform
        self.feature_transform = feature_transform
        if self.point_transform:
            self.stn = STN3d(channel)
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        if self.point_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_cls.py
class PointNetCls(nn.Module):
    def __init__(self, k=40, normal_channel=True):
        super(PointNetCls, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat


class PointNetClsLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetClsLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


# Adapted from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


# Adapted from: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_msg.py
class PointNet2ClsMsg(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(PointNet2ClsMsg, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        return x, l3_points


class PointNet2ClsMsgLoss(nn.Module):
    def __init__(self):
        super(PointNet2ClsMsgLoss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


class PointNet2Encoder(nn.Module):
    def __init__(self, normal_channel=False):
        super().__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        # self.sa1 = PointNetSetAbstractionMsg(
        #     512, [0.01, 0.02, 0.04], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        # )
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.02, 0.04, 0.08], [32, 64, 128], in_channel, [[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # l1_xyz, l1_points = self.sa1(xyz, norm)
        # l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_xyz, l2_points = self.sa2(xyz, norm)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

        return x, l3_points


class HandVerts2ManoAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_encoder = PointNetEncoder()
        # 9 + 6 + 3
        self.mlp = MLP([1024, 512, 256, 18])

    def forward(self, x):
        feat, _, _ = self.hand_encoder(x)
        mano = self.mlp(feat)
        return mano


class HandVerts2ManoCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_encoder = PointNetEncoder()
        self.hand_encoder = PointNetEncoder()
        self.cvae = VAE(
            encoder_layer_sizes=[1024, 256, 512, 256],
            latent_size=256,
            decoder_layer_sizes=[256, 512, 256, 18],
            conditional=True,
            condition_size=1024,
        )

    def forward(self, obj_pts, mano_verts):
        obj_feat, _, _ = self.obj_encoder(obj_pts)
        hand_feat, _, _ = self.hand_encoder(mano_verts)

        recon_x, means, log_var, z = self.cvae(hand_feat, obj_feat)

        return recon_x, means, log_var, z

    def sample(self, obj_pts):
        obj_feat, _, _ = self.obj_encoder(obj_pts)

        z = torch.randn([obj_feat.shape[0], self.cvae.latent_size], device=obj_feat.device)

        recon_x = self.cvae.inference(z, obj_feat)
        return recon_x


class Mano2ManoAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP([51, 256, 512, 256, 51])

    def forward(self, x):
        mano = self.mlp(x)
        return mano


class Mano2ManoCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_encoder = PointNetEncoder()
        # self.hand_encoder =  MLP([51, 256, 512, 256, 51])
        self.cvae = VAE(
            encoder_layer_sizes=[51, 256, 512, 256],
            latent_size=256,
            decoder_layer_sizes=[256, 512, 256, 51],
            conditional=True,
            condition_size=1024,
        )

    def forward(self, obj_pts, mano_param):
        obj_feat, _, _ = self.obj_encoder(obj_pts)

        recon_x, means, log_var, z = self.cvae(mano_param, obj_feat)

        return recon_x, means, log_var, z


class SeqManoVerts2ManoCVAEPos(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_encoder = PointNetEncoder()
        self.hand_encoder = PointNetEncoder()
        self.cvae = ImplicitCVAE(
            condition_size=1024,
            encoder_layer_sizes=[1024, 256, 256],
            latent_size=256,
            decoder_layer_sizes=[256, 512, 256, 15],
            coord_size=3,
        )

    def forward(self, obj_pts, mano_verts, query_coords):
        batch_size, seq_size = mano_verts.shape[0:2]
        obj_feat, _, _ = self.obj_encoder(obj_pts)
        hand_feat, _, _ = self.hand_encoder(mano_verts.reshape(-1, 3, 778))
        hand_feat = hand_feat.reshape(batch_size, -1, hand_feat.shape[-1])
        fused_feat, _ = torch.max(hand_feat, dim=1)

        obj_feat = obj_feat.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, obj_feat.shape[-1])
        fused_feat = fused_feat.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, fused_feat.shape[-1])

        recon_x, means, log_var, z = self.cvae(fused_feat, obj_feat, query_coords.reshape(-1, query_coords.shape[-1]))

        recon_x = recon_x.reshape(batch_size, seq_size, -1)

        return recon_x, means, log_var, z

    def sample(self, obj_pts, query_coords):
        batch_size, seq_size = query_coords.shape[0:2]
        assert batch_size == 1
        obj_feat, _, _ = self.obj_encoder(obj_pts)

        obj_feat = obj_feat.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, obj_feat.shape[-1])

        recon_x = self.cvae.sample(obj_feat, query_coords.reshape(-1, query_coords.shape[-1]))

        recon_x = recon_x.reshape(batch_size, seq_size, -1)

        return recon_x


class SeqManoVerts2ManoCVAET(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_encoder = PointNetEncoder()
        self.hand_encoder = PointNetEncoder()
        self.bottle_neck = MLP([1024, 64])
        self.cvae = ImplicitCVAE(
            condition_size=1024,
            encoder_layer_sizes=[int(64 * 20), 256, 256],
            latent_size=256,
            decoder_layer_sizes=[256, 512, 256, 18],  # rot6d: 6, pose_m: 9, xyz: 3
            coord_size=1,
        )

    def forward(self, obj_pts, mano_verts, query_coords):
        batch_size, seq_size = mano_verts.shape[0:2]
        obj_feat, _, _ = self.obj_encoder(obj_pts)
        hand_feat, _, _ = self.hand_encoder(mano_verts.reshape(-1, 3, 778))
        hand_feat = self.bottle_neck(hand_feat)
        fused_feat = hand_feat.reshape(batch_size, -1, hand_feat.shape[-1]).reshape(batch_size, -1)

        obj_feat = obj_feat.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, obj_feat.shape[-1])
        fused_feat = fused_feat.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, fused_feat.shape[-1])

        recon_x, means, log_var, z = self.cvae(fused_feat, obj_feat, query_coords.reshape(-1, query_coords.shape[-1]))

        recon_x = recon_x.reshape(batch_size, seq_size, -1)

        return recon_x, means, log_var, z

    def sample(self, obj_pts, query_coords):
        batch_size, seq_size = query_coords.shape[0:2]
        assert batch_size == 1

        obj_feat, _, _ = self.obj_encoder(obj_pts)

        obj_feat = obj_feat.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, obj_feat.shape[-1])

        recon_x = self.cvae.sample(obj_feat, query_coords.reshape(-1, query_coords.shape[-1]))

        recon_x = recon_x.reshape(batch_size, seq_size, -1)

        return recon_x


class SeqAllegroVerts2AllegroCVAET(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_encoder = PointNetEncoder()
        self.hand_encoder = PointNetEncoder()
        self.bottle_neck = MLP([1024, 64])
        self.cvae = ImplicitCVAE(
            condition_size=1024,
            encoder_layer_sizes=[int(64 * 20), 256, 256],
            latent_size=256,
            decoder_layer_sizes=[256, 512, 256, 25],  # xyz: 3, rot6d: 6, qpos: 16
            coord_size=1,
        )

    def forward(self, obj_pts, allegro_verts, query_coords):
        batch_size, seq_size, _, verts_num = allegro_verts.shape
        obj_feat, _, _ = self.obj_encoder(obj_pts)
        hand_feat, _, _ = self.hand_encoder(rearrange(allegro_verts, "b n d v -> (b n) d v"))
        hand_feat = self.bottle_neck(hand_feat)
        fused_feat = rearrange(hand_feat, "(b n) d -> b (n d)", b=batch_size)
        obj_feat = repeat(obj_feat, "b d -> (b n) d", n=seq_size)
        fused_feat = repeat(fused_feat, "b d -> (b n) d", n=seq_size)
        query_coords = rearrange(query_coords, "b n 1 -> (b n) 1")

        recon_x, means, log_var, z = self.cvae(fused_feat, obj_feat, query_coords)

        recon_x = rearrange(recon_x, "(b n) d -> b n d", b=batch_size)

        return recon_x, means, log_var, z


class SeqAllegroQpos2AllegroCVAET(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_encoder = PointNetEncoder()
        # self.obj_encoder = PointNet2Encoder()
        self.bottle_neck = MLP([22, 256, 256, 64])
        self.cvae = ImplicitCVAE(
            condition_size=1024,
            encoder_layer_sizes=[int(64 * 20), 256, 256],
            latent_size=256,
            decoder_layer_sizes=[256, 512, 256, 25],  # xyz: 3, rot6d: 6, qpos: 16
            coord_size=1,
            use_trans_mlp=False,
        )

    def forward(self, obj_pts, allegro_qpos, query_coords):
        batch_size, seq_size, _ = allegro_qpos.shape
        obj_feat, _, _ = self.obj_encoder(obj_pts)
        # obj_feat, _ = self.obj_encoder(obj_pts)
        # hand_feat, _, _ = self.hand_encoder(rearrange(allegro_verts, "b n d v -> (b n) d v"))
        hand_feat = self.bottle_neck(allegro_qpos[:, 5:, :])  # first 5 frames are not used for feature extraction
        fused_feat = rearrange(hand_feat, "b n d -> b (n d)", b=batch_size)
        obj_feat = repeat(obj_feat, "b d -> (b n) d", n=seq_size)
        fused_feat = repeat(fused_feat, "b d -> (b n) d", n=seq_size)
        query_coords = rearrange(query_coords, "b n 1 -> (b n) 1")

        recon_x, means, log_var, z = self.cvae(fused_feat, obj_feat, query_coords)

        recon_x = rearrange(recon_x, "(b n) d -> b n d", b=batch_size)

        return recon_x, means, log_var, z

    def sample(self, obj_pts, query_coords, return_code=False):
        batch_size, seq_size = query_coords.shape[0:2]
        assert batch_size == 1

        obj_feat, _, _ = self.obj_encoder(obj_pts)

        obj_feat = obj_feat.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, obj_feat.shape[-1])

        if not return_code:
            recon_x = self.cvae.sample(obj_feat, query_coords.reshape(-1, query_coords.shape[-1]), return_code=False)
        else:
            recon_x, code = self.cvae.sample(
                obj_feat, query_coords.reshape(-1, query_coords.shape[-1]), return_code=True
            )

        recon_x = recon_x.reshape(batch_size, seq_size, -1)

        if not return_code:
            return recon_x
        else:
            return recon_x, code

    def decode(self, obj_pts, query_coords, z):
        batch_size, seq_size = query_coords.shape[0:2]
        assert batch_size == 1

        obj_feat, _, _ = self.obj_encoder(obj_pts)

        obj_feat = obj_feat.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, obj_feat.shape[-1])
        z = z.unsqueeze(1).expand(-1, seq_size, -1).reshape(-1, z.shape[-1])

        recon_x = self.cvae.decode(z, obj_feat, query_coords.reshape(-1, query_coords.shape[-1]))

        recon_x = recon_x.reshape(batch_size, seq_size, -1)

        return recon_x


class ImplicitCVAE(nn.Module):
    def __init__(
        self,
        encoder_layer_sizes,
        latent_size,
        decoder_layer_sizes,
        condition_size=1024,
        coord_size=3,
        use_trans_mlp=False,
    ):
        super().__init__()
        self.latent_size = latent_size
        encoder_layer_sizes[0] += condition_size
        self.encoder_mlp = MLP(encoder_layer_sizes, last_actv=True)
        self.linear_means = nn.Linear(encoder_layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(encoder_layer_sizes[-1], latent_size)

        self.use_trans_mlp = use_trans_mlp
        if not use_trans_mlp:
            decoder_layer_sizes = [latent_size + condition_size + coord_size] + decoder_layer_sizes
            self.decoder_mlp = MLP(decoder_layer_sizes, last_actv=False)
        else:
            decoder_layer_sizes = [latent_size + coord_size] + decoder_layer_sizes
            raise NotImplementedError("TransMlp is not implemented yet")
            # self.trans_mlp = TransMlp(dtoken_dim=condition_size, mlp_layer_dims=decoder_layer_sizes[:-1])
            self.last_relu = nn.ReLU()
            self.last_fc = nn.Linear(decoder_layer_sizes[-2], decoder_layer_sizes[-1])

    def forward(self, x, c, q):
        batch_size = x.size(0)

        means, log_var = self.encode(x, c)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_size], device=means.device)
        z = eps * std + means

        recon_x = self.decode(z, c, q)

        return recon_x, means, log_var, z

    def encode(self, x, c):
        x = torch.cat((x, c), dim=-1)
        x = self.encoder_mlp(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars

    def decode(self, z, c, q):
        if not self.use_trans_mlp:
            z = torch.cat((z, c, q), dim=-1)
            x = self.decoder_mlp(z)
        else:
            c = c.view(-1, 20, c.shape[-1])[:, 0, :]  ## HACK: c (b _), current code provides redundant c
            c = c.unsqueeze(1)  ## HACK: c (b n _)
            z = z.view(-1, 20, z.shape[-1])  ## HACK: z (b T _)
            q = q.view(-1, 20, q.shape[-1])  ## HACK: q (b T _)

            hypo_mlps = self.trans_mlp(c)  # b hypo_mlps
            x = hypo_mlps(torch.cat((z, q), dim=-1))  # (b T _)
            x = self.last_fc(self.last_relu(x))

            x = x.view(-1, x.shape[-1])  ## HACK: to (bT _)
        return x

    def sample(self, c, q, return_code=False):
        z = torch.randn([1, self.latent_size], device=c.device)
        x = self.decode(z.expand((c.shape[0], -1)), c, q)
        if not return_code:
            return x
        else:
            return x, z
