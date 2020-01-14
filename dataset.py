# 批量读取文件夹下的图片并保存在一个四维数组中
import numpy
import os
from PIL import Image  # 导入Image模块
from pylab import *  # 导入savetxt模块


def get_imlist(path):  # 此函数读取特定文件夹下的jpg格式图像，返回图片所在路径的列表
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


c = get_imlist(r"C:\Users\wang\Desktop\DeepLearning\faces")  # r""是防止字符串转译
# print(c)  # 这里以list形式输出jpg格式的所有图像（带路径）
d = len(c)  # 这可以以输出图像个数，如果你的文件夹下有698张图片，那么d为698
print("图片个数：", d)


def image_count():
    return len(c)


def print_dataset(index):
    data = numpy.empty((2000, 96, 96, 3))  # 建立d*（299,299,3）的矩阵
    i = 0
    while i < 2000:
        img = Image.open(c[index])  # 打开图像
        # img_ndarray=numpy.asarray(img)
        img_ndarray = numpy.asarray(img, dtype='float32') / 255  # 将图像转化为数组并将像素转化到0-1之间
        # print(img_ndarray.shape)
        data[i] = img_ndarray  # 将图像的矩阵形式保存到data中
        i = i + 1
        index += 1
    print("data.shape:", data.shape)
    print(index)
    return data, index
