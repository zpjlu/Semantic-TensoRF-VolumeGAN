# python3.7
"""Contains utility functions for rendering."""
import torch
import torch.nn.functional as F
def normalize_vecs(vectors):
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def truncated_normal(tensor, mean=0, std=1):
    """
    Samples from truncated normal distribution.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def get_grid_coords(points, bounds):
    """ transform points from the world coordinate to the volume coordinate
    pts: batch_size, num_point, 3
    bounds: 2, 3
    """
    # normalize the points
    bounds = bounds[None]
    min_xyz = bounds[:, :1]
    points = points - min_xyz
    # convert the voxel coordinate to [-1, 1]
    size = bounds[:, 1] - bounds[:, 0]
    points = (points / size[:, None]) * 2 - 1
    return points

def grid_sample_2d(image, optical):
    """grid sample images by the optical in 3D format
    image: batch_size, channel, H, W
    optical: batch_size, H, W, 2
    """
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]


    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);

    with torch.no_grad():
        ix_lb = torch.floor(ix);
        iy_lb = torch.floor(iy);
 

        ix_rb = ix_lb + 1;
        iy_rb = iy_lb;
  

        ix_lt = ix_lb;
        iy_lt = iy_lb + 1;


        ix_rt = ix_lb + 1;
        iy_rt = iy_lb + 1;
    
    lb = (ix - ix_lb) * (iy -iy_lb);
    rb = (ix_rb - ix) * (iy -iy_rb);
    lt = (ix - ix_lt) * (iy_lt -iy);
    rt = (ix_rt - ix) * (iy_rt -iy);

    with torch.no_grad():
        torch.clamp(ix_lb, 0, IW - 1, out=ix_lb)
        torch.clamp(iy_lb, 0, IH - 1, out=iy_lb)

        torch.clamp(ix_rb, 0, IW - 1, out=ix_rb)
        torch.clamp(iy_rb, 0, IH - 1, out=iy_rb)

        torch.clamp(ix_lt, 0, IW - 1, out=ix_lt)
        torch.clamp(iy_lt, 0, IH - 1, out=iy_lt)

        torch.clamp(ix_rt, 0, IW - 1, out=ix_rt)
        torch.clamp(iy_rt, 0, IH - 1, out=iy_rt)




    image = image.view(N, C, IH * IW)

    lb_val = torch.gather(image, 2,(iy_lb * IW + ix_lb).long().view(N, 1, H * W).repeat(1, C, 1))
    rb_val = torch.gather(image, 2,(iy_rb * IW + ix_rb).long().view(N, 1, H * W).repeat(1, C, 1))
    lt_val = torch.gather(image, 2,(iy_lt * IW + ix_lt).long().view(N, 1, H * W).repeat(1, C, 1))
    rt_val = torch.gather(image, 2,(iy_rt * IW + ix_rt).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (lb_val.view(N, C,  H, W) * lb.view(N, 1, H, W) +
               rb_val.view(N, C,  H, W) * rb.view(N, 1, H, W) +
               lt_val.view(N, C,  H, W) * lt.view(N, 1, H, W) +
               rt_val.view(N, C,  H, W) * rt.view(N, 1, H, W)
               )

    return out_val

def grid_sample_1d(image, optical):
    """grid sample images by the optical in 3D format
    image: batch_size, channel, W
    optical: batch_size, W, 1
    """
    N, C, IW = image.shape
    _, W, _ = optical.shape

    ix = optical[..., 0]



    ix = ((ix + 1) / 2) * (IW - 1);

    with torch.no_grad():
        ix_l = torch.floor(ix);
        ix_r = ix_l + 1;

    
    l = (ix - ix_l);
    r = (ix_r - ix);

    with torch.no_grad():
        torch.clamp(ix_l, 0, IW - 1, out=ix_l)
        torch.clamp(ix_r, 0, IW - 1, out=ix_r)

    l_val = torch.gather(image, 2,ix_l.long().view(N, 1, W).repeat(1, C, 1))
    r_val = torch.gather(image, 2,ix_r.long().view(N, 1, W).repeat(1, C, 1))


    out_val = (l_val.view(N, C,  W) * l.view(N, 1, W) + r_val.view(N, C,  W) * r.view(N, 1, W))

    return out_val

def interpolate_feature(points, plane, line, bounds):
    """
    points: batch_size, num_point, 3
    volume: batch_size, 3, num_channel, h, w
    bounds: 2, 3
    """
    grid_coords = get_grid_coords(points, bounds)

    # point_features = F.grid_sample(volume,
    #                                grid_coords,
    #                                padding_mode='zeros',
    #                                align_corners=True)
    # point_features = grid_sample_3d(volume, grid_coords)
    # point_features = point_features[:, :, 0, 0]
    matMode = [[0,1], [0,2], [1,2]]
    vecMode =  [2, 1, 0]
    app_feature = []
    bs=grid_coords.shape[0]
    coordinate_plane = torch.stack((grid_coords[..., matMode[0]], grid_coords[..., matMode[1]], grid_coords[..., matMode[2]])).detach().view(bs, 3, -1, 1, 2)
    coordinate_line = torch.stack((grid_coords[..., vecMode[0]], grid_coords[..., vecMode[1]], grid_coords[..., vecMode[2]])).detach().view(bs, 3, -1, 1)

    plane_coef_point,line_coef_point = [],[]
    for idx_plane in range(3):
        plane_coef_point.append(grid_sample_2d(plane[:,idx_plane], coordinate_plane[:,idx_plane])[...,0])
        line_coef_point.append(grid_sample_1d(line[:,idx_plane], coordinate_line[:,idx_plane]))
    plane_coef_point, line_coef_point = torch.cat(plane_coef_point,dim=1), torch.cat(line_coef_point,dim=1)

    return plane_coef_point * line_coef_point