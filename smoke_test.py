import torch

from arch import CTKANLight, CTKANMax


def main():
    x = torch.randn(1, 3, 64, 64)
    settings = dict(
        num_classes=1,
        input_channels=3,
        deep_supervision=False,
        img_size=64,
        embed_dims=[32, 48, 64],
        CtmTicks=1,
        CtmMemory=5,
        CtmDaction=64,
        CtmDout=64,
        CtmNself=8,
        CtmLinearLayout="first",
        CtmPriorMode="factorized",
    )
    for model_type in (CTKANLight, CTKANMax):
        output = model_type(**settings).eval()(x)
        print(model_type.__name__, tuple(output.shape))


if __name__ == "__main__":
    main()
