def conv_reverse(conv_param, coordinate):
    """
    conv2d(kernel_size,stride,padding,dilation)の逆関数を実装する。入力テンソル[1,1,m,n]があった時、出力テンソルの[1,1,y,x]の位置に対応する入力テンソルの範囲[1,1,y_1:y_2, x_1:x_2]を計算する。
    Args:
        conv2d_param: 3つの要素を持つリスト。リストは[kernel_size, stride, padding]の順で要素を持つ。
        coordinate: 2つの要素を持つリスト。リストは[x,y]の順で要素を持つ。
    """
    # # 入力テンソルを作成
    # input_tensor = torch.randn(1, 1, 5, 5)

    # # 畳み込み層を定義
    # conv_layer = Conv2d(
    #     in_channels=1,
    #     out_channels=1,
    #     kernel_size=kernel_size,
    #     stride=stride,
    #     padding=padding,
    # )

    # # 畳み込みを実行
    # output_tensor = conv_layer(input_tensor)

    kernel_size, stride, padding, dilation = conv_param
    if isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
        kernel_y, kernel_x = kernel_size
    elif isinstance(kernel_size, int) or len(kernel_size) == 1:
        kernel_y, kernel_x = kernel_size, kernel_size
    else:
        raise ValueError("kernel_sizeはint型か、2要素のリスト型で指定してください。")

    x, y = coordinate

    # 出力テンソルの位置 (x,y) に対応する入力範囲を計算
    start_x = x * stride - padding
    start_y = y * stride - padding

    if dilation == 1:
        end_x = start_x - 1 + kernel_x
        end_y = start_y - 1 + kernel_y
    else:
        end_x = start_x - 1 + dilation * (kernel_x - 1) + 1
        end_y = start_y - 1 + dilation * (kernel_y - 1) + 1

    # print(f"入力範囲: ({start_x}, {start_y}, {end_x}, {end_y})")
    # print("対応する入力テンソルの範囲:")
    # print(input_tensor[0, 0, start_y : end_y + 1, start_x : end_x + 1])
    return [start_x, start_y, end_x, end_y]


def calc_max_feature_size_for_conv_reverse(conv_param, max_feature_size):

    padding = conv_param[2]
    xyxy1 = conv_reverse(conv_param, max_feature_size[:2])
    xyxy2 = conv_reverse(conv_param, max_feature_size[2:])
    next_xyxy = xyxy1[:2] + [i - padding for i in xyxy2[2:]]
    return next_xyxy


def scale_xyxy(xyxy, max_feature_size):
    x1, y1, x2, y2 = xyxy
    mx1, my1, mx2, my2 = max_feature_size
    next_xyxy = [x1, y1, min(x2, mx2), min(y2, my2)]
    return next_xyxy


def net_reverse(conv_params, coordinate, max_feature_size):
    """
    Args:
        conv_params: Nx3のリスト。[kernel_size, stride, padding]がN層にわたって並んでいる。[0,:]はNNの最初の層で、[-1,:]はNNの最後の層に対応する。
        coordinates: 4つの要素を持つリスト。リストは[y1, x1, y2, x2]の順で要素を持つ。y1x1y2x2は入力テンソル上の座標を特定したい特徴量の範囲を指定する。
    """

    rev_conv_params = conv_params[::-1]
    next_coordinates = conv_reverse(rev_conv_params[0], coordinate)
    next_max_feature_size = calc_max_feature_size_for_conv_reverse(
        rev_conv_params[0], max_feature_size
    )
    next_coordinates = scale_xyxy(next_coordinates, next_max_feature_size)

    for cp_i in range(1, len(rev_conv_params)):
        print(f"{cp_i}番目の逆畳み込み-" + str_params(rev_conv_params[cp_i]))
        next_max_feature_size = calc_max_feature_size_for_conv_reverse(
            rev_conv_params[cp_i], next_max_feature_size
        )
        xyxy1 = conv_reverse(rev_conv_params[cp_i], next_coordinates[:2])
        xyxy2 = conv_reverse(rev_conv_params[cp_i], next_coordinates[2:])

        next_coordinates = xyxy1[:2] + xyxy2[2:]
        next_coordinates = scale_xyxy(next_coordinates, next_max_feature_size)
        # print("最大" + str_xyxy(next_max_feature_size))
        print(str_xyxy(next_coordinates))

    return next_coordinates


def generate_params(kernel_size, stride=1, padding=0, dilation=1):
    return (kernel_size, stride, padding, dilation)


def str_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return f"入力範囲: ({x1}, {y1}, {x2}, {y2})"


def str_params(params):
    return f"kernel_size: {params[0]}, stride: {params[1]}, padding: {params[2]}, dilation: {params[3]}"
