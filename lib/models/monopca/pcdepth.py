import torch


def pc_depth(size3d, boxes, ry, img_sizes, calibs):
    """PCDepth: geometry-based depth prior for each query.

        z = H * f / h_2d + 0.5 * (L * |sin(ry)| + W * |cos(ry)|)

    The first term is the pinhole projection of the 3D object height H onto
    the 2D box height h_2d (focal length f). The second term is a yaw-dependent
    correction that moves the visible front face back to the box center.

    Args:
        size3d:    [B, Q, 3] predicted 3D size (H, W, L)
        boxes:     [B, Q, 6] normalized (cx, cy, l, r, t, b) boxes
        ry:        [B, Q]    predicted yaw
        img_sizes: [B, 2]    input image size (W, H)
        calibs:    [B, 3, 4] camera projection matrices
    Returns:
        [B, Q] depth prior
    """
    box2d_height_norm = boxes[:, :, 4] + boxes[:, :, 5]
    box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
    # GeoDepth f * H / h_2d: depth of the visible front surface
    geo_depth = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

    L = size3d[:, :, 2]  # length
    W = size3d[:, :, 1]  # width
    # half-extent of the box footprint along the optical axis
    delta = 0.5 * (L * torch.abs(torch.sin(ry)) + W * torch.abs(torch.cos(ry)))

    return geo_depth + delta
