import torch
import torch.nn.functional as F

def interpolate(tensor, output_dim):
    input_dim = tensor.size(1)
    
    tensor = tensor.unsqueeze(1)
    
    interpolated_tensor = F.interpolate(tensor, size=output_dim, mode='linear', align_corners=True)
    
    return interpolated_tensor.squeeze(1)

tensor = torch.rand(1, 256, device='cuda')
output_dim = 512
result = interpolate(tensor, output_dim)
result.shape, result.device
