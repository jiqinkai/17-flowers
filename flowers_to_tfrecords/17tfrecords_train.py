import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = '/home/zhaozhao/flowers_to_tfrecords/train/'
classes = {'flowerS1', 'flowerS2','flowerS3','flowerS4','flowerS5','flowerS6','flowerS7','flowerS8',
           'flowerS9','flowerS10','flowerS11','flowerS12','flowerS13','flowerS14','flowerS15','flowerS16','flowerS17'}
writer = tf.python_io.TFRecordWriter("train.tfrecords")

for index, name in enumerate(classes):
    class_path = cwd + name + '/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name

        img = Image.open(img_path)
        img = img.resize((128, 128))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())

writer.close()