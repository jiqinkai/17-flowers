from __future__ import absolute_import,division,print_function

import numpy as np
import tensorflow as tf
import time
from os import walk
from os.path import join
from scipy.misc import imread,imresize

DATA_DIR='home/zhaozhao/flowers_to_tfrecords/jpg/'

IMG_hight=227
IMG_width=227
IMG_channels=3
NUM_train=952
NUM_test=408


def read_images(path):
    filenames=next(walk(path))[2]  #遍历目录
    num_files=len(filenames)
    images=np.zeros((num_files,IMG_hight,IMG_width,IMG_channels),dtype=np.uint8)  #遍历所有的图片，将图片热死则到[227,227,3]
    labels=np.zeros((num_files,),dtype=np.uint8)
    f=open('label.txt')
    lines=f.readlines()

    for i,filename in enumerate(filenames):
        img=imread(join(path,filename))
        img=imresize(img,(IMG_hight,IMG_width))
        images[i]=img
        labels[i]=int(lines[i])
    f.close()
    return images,labels

def _int64_feature(value):  #生成整数型的属性
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):  #生成字符串型的属性
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(images,labels,name):
    num=images.shape[0]        #获取转换成TFRecord的数目
    filename=name+'.tfrecords'
    print('Writting',filename)
    writer =tf.python_io.TFRecordWriter(filename)
    for i in range(num):
        img_raw=images[i].tostring()  #将图像矩阵转化为一个字符串
        example=tf.train.Example(features=tf.train.Feature(features={     #将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
            'label':_int64_feature(int(labels[i])),
            'image_raw=':_bytes_feature(img_raw)
        }))
        writer.write(example.SerializeToString())
        writer.close()
        print('Writting end')

def main(argv):
    print('reading images begin')
    start_time=time.time()
    train_images,train_labels=read_images(DATA_DIR)
    duration=time.time()-start_time
    print("reading images end , cost %d sec"%duration)

    test_images=train_images[:NUM_tset,:,:,:]
    test_labels=train_labels[:NUM_test]
    train_images=train_images[NUM_test:,:,:,:]
    train_labels=train_labels[NUM_test:]

    print('convert to tfrecords begin')
    start_time=time.time()
    convert(train_images,train_labels,'train')
    convert((validation_images,validation_labels,'validation'))
    duration=time.time()-start_time
    print('convert to tfrecords end, cost %d sec'%duration)

#if __name__=="__main__":

    #tf.app.run()
