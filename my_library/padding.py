def my_padding(src, pad_shape, pad_type='zero padding'):
    (h, w) = src.shape  # 그림의 사이즈
    (p_h, p_w) = pad_shape  # 늘리고자하는 사이즈
    pad_img = np.zeros((h+2*p_h, w+2*p_w))
    pad_img[p_h:p_h+h, p_w:p_w+w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #up
        for i in range(p_h):
            for j in range(w):
                pad_img[i][j + p_w] = src[0][j]
        #down
        for i in range(h, h + p_h):
            for j in range(w):
                pad_img[i + p_h][j + p_w] = src[h-1][j]
        #left
        for i in range(h + 2*p_h):
            for j in range(p_w):
                pad_img[i][j] = pad_img[i][p_w]
        #right
        for i in range(h + 2*p_h):
            for j in range(w, w + p_w):
                pad_img[i][j + p_w] = pad_img[i][w + p_w - 1]

    else:
        print('zero padding')

    return pad_img