import numpy as np
from PIL import Image
import pickle

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def imgreshape(x,size):
    new_x = []
    # 1.获取图片的通道数以及高宽
    N,C,H,W = x.shape
    # 2.判断通道数
    if C != 1:
        for image in x:
            image_hwc = image.transpose(1,2,0)
            arr2img = Image.fromarray(np.uint8(image_hwc))
            arr2img_1 = arr2img.resize(size,Image.ANTIALIAS)
            img2arr = np.array(arr2img_1)
            imgarr = img2arr.transpose(2,0,1)
            new_x.append(imgarr)
    else:
        for image in x:
            image = image[0]
            arr2img = Image.fromarray(image)
            arr2img_1 = arr2img.resize(size,Image.ANTIALIAS)
            img2arr = np.array(arr2img_1)
            imgarr = img2arr.reshape(1,size[0],size[1])
            new_x.append(imgarr)
    new_x = np.array(new_x)


    return new_x

"""
保存模型
"""
def save_model(model,file_name="model.pkl"):
    with open(file_name,"wb") as f:
        pickle.dump(model,f)

"""
加载模型
"""
def load_model(file_name="model.pkl"):
    with open(file_name,"rb") as f:
        return pickle.load(f)