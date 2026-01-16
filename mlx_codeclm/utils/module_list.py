"""Simple ModuleList replacement for MLX."""

import mlx.nn as nn


class ModuleList(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self._modules_list = list(modules)
        for idx, module in enumerate(self._modules_list):
            setattr(self, f"m{idx}", module)

    def __getitem__(self, idx):
        return self._modules_list[idx]

    def __iter__(self):
        return iter(self._modules_list)

    def __len__(self):
        return len(self._modules_list)
