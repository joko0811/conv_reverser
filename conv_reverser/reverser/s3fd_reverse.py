import math
from .conv_reverse import generate_params, net_reverse
from .vgg_reverse import generate_vgg_params


def s3fd_feature_num_converter(feature_num, feat_sizes):
    """
    Args:
        feature_num: int
    Returns:
        target_layer_num, coordinate: int, list
    """
    s3fd_magical_feature_list = feat_sizes
    s3fd_magical_feature_numbers = [
        s3fd_magical_feature_list[i][0] * s3fd_magical_feature_list[i][1]
        for i in range(len(s3fd_magical_feature_list))
    ]

    total = -1
    offset = -1
    x, y = -1, -1
    for i in range(len(s3fd_magical_feature_list)):
        if feature_num <= total + s3fd_magical_feature_numbers[i]:
            target_layer_num = i
            offset = feature_num - total
            x = int(offset % s3fd_magical_feature_list[i][1])
            y = math.floor(offset / s3fd_magical_feature_list[i][1])
            break
        total += s3fd_magical_feature_numbers[i]
    if offset == -1 or y == -1 or x == -1:
        raise ValueError(
            f"feature_num must be less than {total}, but got {feature_num}."
        )
    max_feature_size = s3fd_magical_feature_list[target_layer_num]
    max_feature_size = [0, 0, max_feature_size[1], max_feature_size[0]]
    return target_layer_num, (x, y), max_feature_size


def s3fd_reverse(target_layer_num, coordinate, max_feature_size):
    # calc vgg conv_params
    vgg_params = generate_vgg_params()
    # calc extra layers conv_params
    extra_layers_params = generate_extra_layers_params()
    # calc multihead conv_params
    multihead_params = generate_multihead_params(
        len(vgg_params), len(extra_layers_params)
    )

    multihead_targets = [15, 22, 29]
    for i in range(len(multihead_params)):
        if i % 2 == 1:
            multihead_targets.append(len(vgg_params) + i)

    target_layer_num = multihead_targets[target_layer_num]

    target_params = []

    if target_layer_num < len(vgg_params):
        # vgg reverse
        target_params = vgg_params[: target_layer_num - 1]
    else:
        # extra layers reverse
        target_params = (
            vgg_params + extra_layers_params[: target_layer_num - len(vgg_params) - 1]
        )

    target_params.append(multihead_params[multihead_targets.index(target_layer_num)])
    # vgg reverse
    x1y1x2y2 = net_reverse(target_params, coordinate, max_feature_size)

    return x1y1x2y2


def generate_multihead_params(vgg_length, extra_layers_length):
    # Of the multiheads, conf is not used because it is not used for source coordinate estimation.
    loc_params = []

    loc_params.append(generate_params(3, padding=1))

    for _ in range(vgg_length):
        loc_params.append(generate_params(3, padding=1))

    for _ in range(2, int(extra_layers_length / 2)):
        loc_params.append(generate_params(3, padding=1))

    return loc_params


def generate_extra_layers_params():
    params = []
    extras_cfg = [256, "S", 512, 128, "S", 256]

    count_s = 0
    prev_e = ""

    is_even_numbered_n = False

    for i, e in enumerate(extras_cfg):
        if prev_e != "S":
            kernel_size = 1 if (count_s % 2) == 0 else 3
            if e == "S":
                params.append(generate_params(kernel_size, stride=2, padding=1))
            else:
                params.append(generate_params(kernel_size))
            is_even_numbered_n = not is_even_numbered_n
            count_s += 1
        prev_e = e

    return params
