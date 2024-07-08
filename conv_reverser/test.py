from reverser import s3fd_reverse

if __name__ == "__main__":
    feature_idx = 169128
    target_layer, (feat_x, feat_y), max_feature_size = (
        s3fd_reverse.s3fd_feature_num_converter(feature_idx)
    )
    print((feat_x, feat_y))
    xyxy = s3fd_reverse.s3fd_reverse(
        target_layer, (feat_x, feat_y), [0, 0] + max_feature_size
    )
    print(xyxy)
