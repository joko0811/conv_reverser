from .conv_reverse import generate_params


def generate_vgg_params():
    """
    output:
        params:
            [
                (kernel_size, stride, padding),
                (kernel_size, stride, padding),
                ...
            ]
    """
    vgg_cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "C",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]

    params = []

    for v in vgg_cfg:
        if (v == "M") or (v == "C"):
            params.append(generate_params(2, stride=2))
        else:
            params.append(generate_params(3, padding=1))

    # conv6 dilation=6
    params.append(generate_params(3, padding=6, dilation=6))
    # conv7
    params.append(generate_params(1))

    return params
