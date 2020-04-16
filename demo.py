
import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sift_flow_torch import SiftFlowTorch
from third_party.flowiz import flowiz
from skimage.io import imread, imsave
device = torch.device("cuda:2")
def find_local_matches(desc1, desc2, kernel_size=9):
    # Computes the correlation between each pixel on desc1 with all neighbors
    # inside a window of size (kernel_size, kernel_size) on desc2. The match
    # vector if then computed by linking each pixel on desc1 with
    # the pixel with desc2 with the highest correlation.
    #
    # This approch requires a lot of memory to build the unfolded descriptor.
    # A better approach is to use the Correlation package from e.g.
    # https://github.com/NVIDIA/flownet2-pytorch/tree/master/networks/correlation_package
    desc2_unfolded = F.unfold(desc2, kernel_size, padding=kernel_size//2)
    desc2_unfolded = desc2_unfolded.reshape(
        1, desc2.shape[1], kernel_size*kernel_size, desc2.shape[2], desc2.shape[3])
    desc1 = desc1.unsqueeze(dim=2)
    correlation = torch.sum(desc1 * desc2_unfolded, dim=1)
    _, match_idx = torch.max(correlation, dim=1)
    hmatch = torch.fmod(match_idx, kernel_size) - kernel_size // 2
    vmatch = match_idx // kernel_size - kernel_size // 2
    matches = torch.cat((hmatch, vmatch), dim=0)
    return matches

# If you run out or memory, try increasing the two variables below.
# Running this example with sift_step_size=2, image_resize_factor=1, and fp16=True
# requires a GPU with at least 8GB
sift_step_size = 2
image_resize_factor = 1

sift_flow = SiftFlowTorch(
    cell_size=2,
    step_size=sift_step_size,
    is_boundary_included=True,
    num_bins=8,
    cuda=True,
    device=device,
    fp16=True,
    return_numpy=False)
imgs = [
    cv2.imread('/projects/grail/xiaojwan/nba2k_flow/2k_frames/NBA2K19_2019.01.31_23.50.52_frame1001579.png'),
    cv2.imread('/projects/grail/xiaojwan/2k_players_mesh_rasterized_noised_camera_sigma_5/NBA2K19_2019.01.31_23.50.52_frame1001579.png')
]
imgs = [cv2.resize(im, (im.shape[1]//image_resize_factor, im.shape[0]//image_resize_factor)) for im in imgs]
print('Warm-up step, will be slow on GPU')
torch.cuda.synchronize()
start = time.perf_counter()
descs = sift_flow.extract_descriptor(imgs)
torch.cuda.synchronize()
end = time.perf_counter()
print('Time: {:.03f} ms'.format((end - start) * 1000))

print('Subsequent runs will be faster (on GPU)')
torch.cuda.synchronize()
start = time.perf_counter()
descs = sift_flow.extract_descriptor(imgs)
torch.cuda.synchronize()
end = time.perf_counter()
print('Time: {:.03f} ms'.format((end - start) * 1000))

print('Descriptor shape:', descs.shape)

flow = find_local_matches(descs[0:1], descs[1:2], 7)


# Show optical flow
flow = flow.permute(1, 2, 0).detach().cpu().numpy()
flow_img = flowiz.convert_from_flow(flow)
imsave('flow.png', flow_img)
