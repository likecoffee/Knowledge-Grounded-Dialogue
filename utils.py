import torch
import torch.nn as nn

PAD_TOKEN = 0
NEAR_INF = 1e20

def cos_sim(tensor_a, tensor_b):
    """
        Args:
            tensor_a : (FloatTensor)    : [batch_size, size]   
            tensor_b : (FloatTensor)    : [batch_size, size]   
        Returns:
            scores :   (FloatTensor)    : [batch_size]
    """
    dot_product = torch.bmm(tensor_a.unsqueeze(1), 
                            tensor_b.unsqueeze(2)).squeeze()
    norm_score = torch.norm(tensor_a, dim = 1) * torch.norm(tensor_b, dim = 1)

    return dot_product / norm_score

def generate_mask_by_length(seq_len, max_len, masked_mode = "mul"):
    """
    Args:
        seq_len (LongTensor):   [batch_size]
        max_len (int)       :   the maximum of seq_len
        masked_mode (str)  :    the mode used to mask position, the detail is:
                                ----------------------------------------------
                                    MODE            UNMASK      MASK
                                    mul             0           -NEAR_INF
                                    add             1           0
                                ----------------------------------------------
    Returns:
        mask (FloatTensor) :    [max_len, batch_size]
    """
    batch_size = seq_len.size(0)
    if masked_mode == "mul":
        mask = torch.zeros(max_len, batch_size, device = seq_len.device)
        mask_idx = -NEAR_INF
    elif masked_mode == "add":
        mask = torch.ones(max_len, batch_size, device = seq_len.device)
        mask_idx = 0
    else:
        raise ValueError("maksked_mode must be 'mul' or 'add'")
    # length_tensor : [max_len, batch_size]
    length_tensor = torch.arange(max_len).unsqueeze(1).expand(max_len, batch_size).to(seq_len.device)
    # seq_len : [max_len, batch_size]
    seq_len = seq_len.unsqueeze(0).expand_as(length_tensor)
    # mask_bool : [max_len, batch_size]
    mask_bool = length_tensor >= seq_len
    mask[mask_bool] = mask_idx 

    return mask
        
def pad_tensor_by_length(tensor, max_len, dim, padding_idx = 0):
    """pad tensor to given length in specific dimension by padding_idx. 
    Args:
        tensor : (Tensor)   : tensor.size(dim) exists
        max_len : (int)     : the maximum length
        dim : (int)         : the dimension should be padded
        padding_idx : (int) : the padding idx used 
    Returns:
        tensor : (Tensor)   : tensor.size(dim) == max_len
    """
    # number of dimension should be greater than dim
    assert tensor.dim() >= dim
    tensor_size = list(tensor.size())
    if tensor.size(dim) < max_len:
        padded_size = max_len - tensor.size(dim)
        tensor_size[dim] = padded_size
        return torch.cat([tensor, tensor.new(*tensor_size).zero_()], dim=dim)
    else:
        return tensor

class Identity(nn.Module):
    def forward(self, x):
        return x
