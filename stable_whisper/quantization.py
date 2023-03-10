import torch
from torch import nn
from whisper.model import Linear, Conv1d, LayerNorm, Whisper


def replace_modules(model: nn.Module, only_linear: bool = False):
    """
    replace Linear/Conv1d/LayerNorm from `whisper.model` with equivalent module in `torch.nn`
    """
    for m in model.__dict__.get('_modules', []):
        module = model.__getattr__(m)
        update = True
        if isinstance(module, Linear):
            model.__setattr__(m, nn.Linear(module.in_features, module.out_features,
                                           bias=module.bias is not None))
        elif not only_linear and isinstance(module, Conv1d):
            model.__setattr__(m, nn.Conv1d(module.in_channels, module.out_channels,
                                           kernel_size=module.kernel_size,
                                           stride=module.stride,
                                           padding=module.padding,
                                           bias=module.bias is not None))
        elif not only_linear and isinstance(module, LayerNorm):
            model.__setattr__(m, nn.LayerNorm(module.normalized_shape[0]))
        else:
            update = False
            replace_modules(module)

        if update:
            model.__getattr__(m).load_state_dict(module.state_dict())


def ptdq_linear(model: "Whisper"):
    """
    apply Dynamic Quantization to Whisper model instance
    """
    model.cpu()
    replace_modules(model, only_linear=True)
    torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=True)
    setattr(model, 'dq', True)
