import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# local transformer
def extract_spatial_layout_feats(xyxy_boxes):
    box_center = torch.stack([(xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2, (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2], dim=1)
    dxdy = box_center.unsqueeze(1) - box_center.unsqueeze(0) # distances
    theta = (torch.atan2(dxdy[...,1], dxdy[...,0]) / np.pi).unsqueeze(-1)
    dis = dxdy.norm(dim=-1, keepdim=True)

    box_area = (xyxy_boxes[:, 2:] - xyxy_boxes[:, :2]).prod(dim=1) # areas
    intersec_lt = torch.max(xyxy_boxes.unsqueeze(1)[...,:2], xyxy_boxes.unsqueeze(0)[...,:2])
    intersec_rb = torch.min(xyxy_boxes.unsqueeze(1)[...,2:], xyxy_boxes.unsqueeze(0)[...,2:])
    overlap = (intersec_rb - intersec_lt).clamp(min=0).prod(dim=-1, keepdim=True)
    union_lt = torch.min(xyxy_boxes.unsqueeze(1)[...,:2], xyxy_boxes.unsqueeze(0)[...,:2])
    union_rb = torch.max(xyxy_boxes.unsqueeze(1)[...,2:], xyxy_boxes.unsqueeze(0)[...,2:])
    union = (union_rb - union_lt).clamp(min=0).prod(dim=-1, keepdim=True)
    spatial_feats = torch.cat([
        dis,  # dx, dy, distance, theta
        overlap/(union+1e-6) # overlap, union, subj, obj
    ], dim=-1)
    return spatial_feats
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
    #return torch.cdist(src, dst)**2
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    #raw_size = idx.size()
    #idx = idx.reshape(raw_size[0], -1)
    #res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    #return res.reshape(*raw_size, -1)

    B,S,K = idx.shape
    B,N,C = points.shape
    points = points.reshape(B*N,C)
    batch_dim = torch.arange(0, B, device = idx.device)
    idx = (batch_dim[:,None] * N +  idx.reshape(B,S*K)).flatten()
    return points[idx].reshape(B,S,K,C)

class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_delta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.box_pos = nn.Linear(2,1)

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
    def forward(self, features,pos,pos_center): # xyz: b*3*n, features: b*n*f
        B,N,C=features.shape
        x_c, y_c, w, h = pos_center.unbind(-1)
        box_coord = torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)],dim=-1)
        dists = torch.zeros((B,N,N))
        for i in range(B):
            dists[i]=torch.squeeze(self.box_pos(extract_spatial_layout_feats(box_coord[i])),-1)
        
        knn_idx = torch.topk(dists, self.k, dim = -1, largest = False, sorted = False).indices
        knn_pos = index_points(pos, knn_idx) # b x n x k x 3
        pre = features # b x n x f
        x=features
        x = self.fc1(features) # b x n x C
        q, k, v = self.w_qs(x), self.w_ks(index_points(x,knn_idx)),self.w_vs(index_points(x,knn_idx))
        #q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q: b x n x C, k: b*n*k*C, v: b*n*k*C
        pos_enc = self.fc_delta(pos[:, :, None] - knn_pos)  # b x n x k x C
        attn = self.fc_gamma(q[:, :, None] - k+pos_enc) # b x n x k x C
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x C
    
        res = (attn * (v+pos_enc)).sum(dim=-2)
        res = self.fc2(res)+pre
        return res
  
class Geometric_Group(nn.Module):
    def __init__(self,
                 num_layers=1,
                 d_points=256,
                 ffn_dim_expansion=1,
                 k=2,
                 ):
        super(Geometric_Group, self).__init__()

        self.d_points = d_points
        self.d_model = d_points * ffn_dim_expansion
        self.k = k
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_points=self.d_points,
                               d_model=self.d_model,
                               k=2,
                               )])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0,pos,pos_center,**kwargs,
                ):

        b, n, c = feature0.shape
        assert self.d_points == c
        feature = None
        for i, layer in enumerate(self.layers):
            feature1 = layer(feature0,pos,pos_center)
            if(i==0):
                feature=feature1
            else:
                feature+=feature1
        feature = feature/self.num_layers
        feature = feature.view(b, n, c).contiguous()
        return feature
