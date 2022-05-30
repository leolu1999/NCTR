import torch
from torch import nn
import torch_scatter as ts
from models.dgmc.models.spline import SplineCNN
from models.dgmc.models.dgmc import DGMC


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def edge_attr(aw, kpts):
    bs = aw.shape[0]
    num_points = aw.shape[2]
    num_neighbor = 8
    awm = torch.mean(aw, dim=1, keepdim=False, out=None)
    edge = torch.topk(awm, num_neighbor, dim=2, largest=True, sorted=True, out=None)
    indices0 = edge.indices.view(1, -1)
    zero = torch.arange(0, awm.shape[1])
    zero = zero.repeat(1, num_neighbor)
    indices1, _ = torch.sort(zero, descending=False, dim=-1)
    indices1 = indices1.repeat(1, bs)
    indices1 = indices1.cuda()
    edge = torch.cat((indices0, indices1), 0)  # 求edge[2,num_neighbor*num_point*bs]
    kpts = kpts.view(-1, 2)
    indices00, indices11 = indices0.squeeze(), indices1.squeeze()
    attr = kpts[indices00, :] - kpts[indices11, :]
    amax, amin = torch.max(attr, 0).values, torch.min(attr, 0).values
    attr = (attr - amin) / (amax - amin)  # 求attr[num_neighbor*num_point*bs,2]
    batch = torch.arange(0, bs).cuda()
    batch = batch.repeat(1,num_points)
    batch, _ = torch.sort(batch, descending=False, dim=-1)
    batch = batch.squeeze()  # 求batch[num_point*bs,]
    return edge, attr, batch


class Attention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [B, N(Num_point1), H, D]
            keys: [B, M(Num_point2), H, D]
            values: [B, M(Num_point2), H, D]
        Returns:
            queried_values: (N, N, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("bnhd,bmhd->bnmh", queries, keys)  # (B, N, M, H)

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)  # (B, N, M, H)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("bnmh,bmhd->bnhd", A, values)  #(B, N, H, D)
        attention_weight = A.permute(0, 3, 1, 2)
        return queried_values.contiguous(), attention_weight


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(EncoderLayer, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.attention = Attention()
        self.merge = nn.Linear(d_model, d_model, bias=True)
        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=True),
            nn.GELU(),
            nn.Linear(d_model*2, d_model, bias=True),
        )
        # norm and dropout
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source):
        """
        Args:
            x (torch.Tensor): [B, C, N(Num_point1)]
            source (torch.Tensor): [B, C, M(Num_point2)]
        """
        bs = x.size(0)
        x, source = x.transpose(1, 2), source.transpose(1, 2)
        x, source = self.norm0(x), self.norm0(source)
        query, key, value = x, source, source  # [N, Num_point, C]
        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)   # [B, N, H, D]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [B, M, H, D]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)  # [B, M, H, D]
        message, attention_weight = self.attention(query, key, value)  # [B, N, H, D]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [B, N, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return message, attention_weight


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(feature_dim, 4) for _ in range(len(layer_names))])
        self.names = layer_names
        self.psi_2 = SplineCNN(64, 64, 2, 2, cat=True, dropout=0.0)
        self.model = DGMC(self.psi_2, num_steps=10)
        self.merge = nn.Sequential(
            nn.Linear(feature_dim+64, feature_dim+64, bias=True),
            nn.GELU(),
            nn.Linear(feature_dim+64, feature_dim, bias=True),
        )
        self.norm1 = nn.LayerNorm(64)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, desc0, desc1, kpts0, kpts1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
                delta0, aw1 = layer(desc0, src0)
                delta1, aw2 = layer(desc1, src1)
                delta0, delta1 = delta0.transpose(1, 2), delta1.transpose(1, 2)
                desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                delta0, aw1 = layer(desc0, src0)
                delta1, aw2 = layer(desc1, src1)
                edge1, attr1, batch1 = edge_attr(aw1, kpts0)
                edge2, attr2, batch2 = edge_attr(aw2, kpts1)
                desc00, desc11 = desc0.transpose(1, 2), desc1.transpose(1, 2)
                desc00, desc11 = desc00.transpose(0, 1).contiguous().view(-1, 256), desc11.transpose(0, 1).contiguous().view(-1, 256) # [bs*num_point,256]
                d0 = self.model(desc00, edge1, attr1, batch1,
                                desc11, edge2, attr2, batch2)  # [B, 64, N]
                d1 = self.model(desc11, edge2, attr2, batch2,
                                desc00, edge1, attr1, batch1)  # [B, 64, N]
                d0, d1 = d0.transpose(1, 2), d1.transpose(1, 2)  # [B, N, 64]
                d0, d1 = self.norm1(d0), self.norm1(d1)
                cat0, cat1 = torch.cat([delta0, d0], dim=2), torch.cat([delta1, d1], dim=2)  # [B, N, C+64]
                delta0, delta1 = self.merge(cat0).transpose(1, 2), self.merge(cat1).transpose(1, 2)  # [B, N, C]
                desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1), torch.cat([bins1, alpha], -1)], 1)

    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class NCTR(nn.Module):
    default_config = {
        'descriptor_dim': 256,
        'weights_path': None,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
        'use_layernorm': False
    }

    def __init__(self, config, keypoint_position_dim=2):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.keypoint_position_dim = keypoint_position_dim

        self.kenc = KeypointEncoder(self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.norm = nn.LayerNorm(self.config['descriptor_dim'])
        self.gnn = AttentionalGNN(self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'], kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        if self.config['weights_path']:
            weights = torch.load(self.config['weights_path'], map_location="cpu")
            if ('ema' in weights) and (weights['ema'] is not None):
                load_dict = weights['ema']
            elif 'model' in weights:
                load_dict = weights['model']
            else:
                load_dict = weights
            self.load_state_dict(load_dict)
            print('Loaded model (\"{}\" weights)'.format(self.config['weights_path']))

    def forward(self, data):
        desc0, desc1 = data['descriptors0'], data['descriptors1']  #[256,batch,num point]
        kpts0, kpts1 = data['keypoints0'], data['keypoints1'] #[1,batch,num point,2]
        bs = desc0.shape[1]
        desc0, desc1 = desc0.permute(1, 2, 0), desc1.permute(1, 2, 0)  #[batch, 256 ,num]
        kpts0 = torch.reshape(kpts0, (bs, -1, 2))  #[bs,num,2]
        kpts1 = torch.reshape(kpts1, (bs, -1, 2))

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
            }

        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        scores0 = torch.transpose(data['scores0'], 0, 1)
        scores1 = torch.transpose(data['scores1'], 0, 1)
        desc0 = desc0 + self.kenc(kpts0, scores0)
        desc1 = desc1 + self.kenc(kpts1, scores1)

        # Multi-layer Transformer network.
        desc0, desc1 = self.norm(desc0.transpose(1, 2)), self.norm(desc1.transpose(1, 2))
        desc0, desc1 = self.gnn(desc0.transpose(1, 2), desc1.transpose(1, 2), kpts0, kpts1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5
        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score, iters=self.config['sinkhorn_iterations'])

        if self.training:
            gt_indexes = data['matches']
            neg_flag = (gt_indexes[:, 1] == -1) | (gt_indexes[:, 2] == -1)
            loss_pre_components = scores[gt_indexes[:, 0], gt_indexes[:, 1], gt_indexes[:, 2]]
            loss_pre_components = torch.clamp(loss_pre_components, min=-100, max=0.0)
            loss_vector = -1 * loss_pre_components
            neg_index, pos_index = gt_indexes[:, 0][neg_flag], gt_indexes[:, 0][~neg_flag]
            # batched_loss = ts.scatter_mean(loss_vector, gt_indexes[:, 0])
            batched_pos_loss, batched_neg_loss = ts.scatter_mean(loss_vector[~neg_flag], pos_index,
                                                                 dim_size=bs), ts.scatter_mean(
                loss_vector[neg_flag], neg_index, dim_size=bs)
            pos_loss, neg_loss = self.config['pos_loss_weight'] * batched_pos_loss.mean(), self.config[
                'neg_loss_weight'] * batched_neg_loss.mean()
            loss = pos_loss + neg_loss
            return loss, pos_loss, neg_loss
        else:
            # Get the matches with score above "match_threshold".
            max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
            indices0, indices1 = max0.indices, max1.indices
            mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
            mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
            zero = scores.new_tensor(0)
            mscores0 = torch.where(mutual0, max0.values.exp(), zero)
            mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
            valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
            valid1 = mutual1 & valid0.gather(1, indices1)
            indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
            indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
            return {
                'matches0': indices0,  # use -1 for invalid match
                'matches1': indices1,  # use -1 for invalid match
                'matching_scores0': mscores0,
                'matching_scores1': mscores1,
            }
