import os
import re
import math
from typing import Any
import torch
import torch.nn as nn
from peft.tuners.lora.layer import Linear
from safetensors.torch import load_file


class TopSingularValue(nn.Module):
    def __init__(self, token_dim, r, dtype):
        super().__init__()
        self.r = r
        self.token_dim = token_dim
        self.weight = nn.Parameter(torch.empty((token_dim, r), dtype=dtype))
        self.rms_norm = nn.RMSNorm([r])
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode="fan_out")

    def forward(self, x):
        weight = x @ self.weight
        weight = self.rms_norm(weight)
        weight = torch.exp(weight)
        return weight


class TopLinear(Linear):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        adapter_name = kwargs["adapter_name"]
        self.lora_lambda = nn.ModuleDict({})
        self.toplora_init(adapter_name)
        
    def toplora_init(self, adapter_name):
        if not self.lora_lambda:
            # first toplora layer being added, add lora_lambda to the list of learnable parameters
            self.adapter_layer_names = self.adapter_layer_names[:] + ("lora_lambda",)
        
        layer = TopSingularValue(
            token_dim=self.in_features,
            r=self.r[adapter_name],
            dtype =self.lora_A[adapter_name].weight.dtype,
            ).to(self.lora_A[adapter_name].weight.device)
        self.lora_lambda[adapter_name] = layer

    def get_delta_weight(self, adapter) -> torch.Tensor:
        raise ValueError("toplora doese not support get_delta_weight.")

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                lora_lambda = self.lora_lambda[active_adapter]
                if not self.use_dora[active_adapter]:
                    # result = result + lora_B(lora_A(dropout(x))) * scaling
                    singular_values = lora_lambda(x)
                    result = result + lora_B(lora_A(dropout(x))*singular_values) * scaling
                else:
                    raise ValueError("toplora doese not support DoRA.")
                
            result = result.to(torch_result_dtype)

        return result


def replace_toplora_linear(
        model, 
        adapter_name="default", 
        init_lora_weights=True
        ):
    """
    Args:
        init_lora_weights: the same as the input of LoraConfig, default: True.
    """

    for name, module in model.named_modules():
        if isinstance(module, Linear):
            # print(name)
            if adapter_name not in module._active_adapter:
                raise ValueError(f"PeftModel does not have '{adapter_name}' adapter.")

            r = module.r[adapter_name]
            lora_alpha = module.lora_alpha[adapter_name]
            lora_dropout = (
                module.lora_dropout[adapter_name].p
                if isinstance(module.lora_dropout[adapter_name], nn.Dropout)
                else 0
            )
            use_rslora = (module.scaling[adapter_name] == (lora_alpha / math.sqrt(r)))

            kwargs = {
                "base_layer": module.base_layer,
                "adapter_name": adapter_name,
                "r": r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "fan_in_fan_out": module.fan_in_fan_out,
                "is_target_conv_1d_layer": module.is_target_conv_1d_layer,
                "init_lora_weights": init_lora_weights,
                "use_rslora": use_rslora,
                "use_dora": module.use_dora[adapter_name],
                "lora_bias": module.lora_bias[adapter_name],
                "kwargs": module.kwargs,
            }

            toplinear = TopLinear(**kwargs)
            parent = model
            submodules = name.split(".")
            for submodule in submodules[:-1]:
                parent = getattr(parent, submodule)
            setattr(parent, submodules[-1], toplinear)


def load_toplora_model(model, path, adapter_name="default"):
    state_dict = load_file(os.path.join(path, "adapter_model.safetensors"))
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = re.sub(r"lora(.+?)\.", r"lora\1."+adapter_name+".", key)
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict, strict=False)
