import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker, matrix_product_state


def tt_decomposition_conv_layer(layer, ranks):
    data = layer.weight.data
    data2D = tl.base.unfold(data, 0)

    first, last = matrix_product_state(data2D, rank=ranks)
    
    first = first.reshape(data.shape[0], ranks, 1, 1)
    last = last.reshape(ranks, data.shape[1], layer.kernel_size[0], layer.kernel_size[1])

    bias = layer.bias is not None
    if layer.bias is None:
        bias = False
    else:
        bias = True
    
    pointwise_s_to_r_layer = torch.nn.Conv2d(
            in_channels=last.shape[1], 
            out_channels=last.shape[0], 
            kernel_size=layer.kernel_size, 
            stride=layer.stride, 
            padding=layer.padding, 
            dilation=layer.dilation, 
            bias=bias)

    pointwise_r_to_t_layer = torch.nn.Conv2d(
            in_channels=first.shape[1], 
            out_channels=first.shape[0], 
            kernel_size=1, 
            stride=1,
            padding=0, 
            dilation=layer.dilation, 
            bias=True)

    if bias: 
        pointwise_r_to_t_layer.bias.data = layer.bias.data
    pointwise_s_to_r_layer.weight.data = last
    pointwise_r_to_t_layer.weight.data = first

    new_layers = [pointwise_s_to_r_layer, pointwise_r_to_t_layer]
    
    return nn.Sequential(*new_layers)


def _decompose(module, factor):
    if isinstance(module, nn.modules.conv.Conv2d):
        conv_layer = module
        rank = max(conv_layer.weight.data.cpu().numpy().shape) // factor
        try:
            module = tt_decomposition_conv_layer(conv_layer, rank)
        except:
            module = module
        return module
    else:
        if len(module._modules) == 0:
            return module
        else:
            for key in module._modules.keys():
                module._modules[key] = _decompose(module._modules[key], factor)
            return module


def decompose(model, factor, savepath='./', include_params_and_factor=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tl.set_backend('pytorch')
    full_model = model.to(device)
    torch.save(full_model.state_dict(), savepath + 'full_model')

    model = _decompose(model, factor)

    if include_params_and_factor:
        nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{nParam}')
        torch.save(model.state_dict(), f'{savepath}decomposed_{factor=}_{nParam=}')
    else:
        torch.save(model.state_dict(), f'{savepath}decomposed')
    
    return model