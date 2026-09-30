from torch import nn

from arch.CTKANLight import CTKANLight, _make_block, _network_settings


class CTKANMax(CTKANLight):
    """Four-block model using adaptive CTKAN at every token stage."""

    def __init__(self, *args, **kwargs):
        settings = _network_settings(kwargs)
        super().__init__(*args, **kwargs)
        dims = settings["embed_dims"]
        self.block1 = nn.ModuleList([_make_block(dims[1], settings, 0)])
        self.dblock2 = nn.ModuleList([_make_block(dims[0], settings, 1)])


__all__ = ["CTKANMax"]
