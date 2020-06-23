import numpy as np
import struct

from PIL import Image
import os

data_file = 'C:/Users/cyx/Desktop/CNN/train-images.idx3-ubyte' #训练集数据路径

# It's 47040016B, but we should set to 47040000B
data_file_size = 47040016  #60000*28*28=47040000
data_file_size = str(data_file_size - 16) + 'B'

data_buf = open(data_file, 'rb').read()

magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)

label_file = 'C:/Users/cyx/Desktop/CNN/train-labels.idx1-ubyte'#训练集数据路径

# It's 60008B, but we should set to 60000B
label_file_size = 60008
label_file_size = str(label_file_size - 8) + 'B'

label_buf = open(label_file, 'rb').read()

magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from(
    '>' + label_file_size, label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

datas_root = 'C:/Users/cyx/Desktop/CNN/train_images' #训练集图片保存路径

if not os.path.exists(datas_root):
    os.mkdir(datas_root)

for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

for ii in range(numLabels):
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    #不足5位补0版本
    file_name = datas_root + os.sep + str(label) + os.sep + \
        str(ii).zfill(5) + '.png'
    img.save(file_name)

