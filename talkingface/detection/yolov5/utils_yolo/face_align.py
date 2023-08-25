import numpy as np
from skimage import transform as trans

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
#<--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

#---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

#-->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

#-->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])

src_map = {
    112: src,
    224: src * 2,
    240: src * 2.142857142857143,
    245: src * 2.1875,
    248: src * 2.214285714285714,
    252: src * 2.25,
    256: src * 2.28571428
}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


# lmk is prediction; src is template
def estimate_norm(lmk,
                  image_size=112,
                  mode='arcface',
                  face_pad=0.0,
                  inference=False,
                  pad_version="v0"):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_m = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        padsize = [int(image_size * face_pad)] * 4
        if (pad_version == "v3_train"):
            padsize[0], padsize[1], padsize[2], padsize[
                3] = padsize[0] * 16 // 12, padsize[0] * 16 // 12, padsize[
                    0] * 8 // 12, padsize[0] * 24 // 12
        if (pad_version == "v3"):
            # padsize[0], padsize[1], padsize[2], padsize[3] = padsize[0] * 9 // 12, padsize[0] * 9 // 12, -padsize[0] * 4 // 12, padsize[0] * 22 // 12
            padsize[0], padsize[1], padsize[2], padsize[3] = int(
                padsize[0] * 344 / 171), int(padsize[0] * 344 / 171), int(
                    padsize[0] * 36 / 171), int(padsize[0] * 652 / 171)

        basepad_ratio = image_size / 256.0 if pad_version == "v3_train" else 0
        ext_left, ext_right, ext_top, ext_down = \
            25 * basepad_ratio + padsize[0], 25 * basepad_ratio + padsize[1], 12.5 * basepad_ratio + padsize[2], 37.5 * basepad_ratio + padsize[3]

        arcface_src_tran = arcface_src.copy()

        arcface_src_tran[:, :, 0] = ext_left + arcface_src[:, :, 0] * (
            image_size - ext_left - ext_right) / 112
        arcface_src_tran[:, :, 1] = ext_top + arcface_src[:, :, 1] * (
            image_size - ext_top - ext_down) / 112

        source = arcface_src_tran
    else:
        source = src_map[112] * image_size / 112.0

    for i in np.arange(source.shape[0]):
        tform.estimate(lmk, source[i])
        m = tform.params[0:2, :]
        results = np.dot(m, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - source[i])**2, axis=1)))
        if error < min_error:
            min_error = error
            min_m = m
            min_index = i
    return min_m, min_index, results
