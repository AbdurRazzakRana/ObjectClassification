import cv2
import os
import glob
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import os, glob, cv2
import sys, argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path = sys.argv[1]
filename = dir_path + '/' + image_path + '/'
path = os.path.join('testing_data', filename, '*g')
files = glob.glob(path)
decisionMartrix = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

for i in files:

    filename = i
    print(i)
    p=0
    className = ""
    for k in i:
        if i[p] =='.':
            break
        if p >= 26:
            className= className + k
        #print(i[p])
        #print(p)
        p = p + 1

    # image_path=sys.argv[1]
    # filename = dir_path +'/'+'test/bag.1.jpg'
    className = className + 's'
    #print(className)
    image_size = 128
    num_channels = 3
    num_classes = 12
    classes = ['bags', 'baskets', 'books', 'chairs', 'keyboards', 'keys', 'laptops', 'mobiles', 'monitors', 'mouses',
               'pens', 'tables']
    images = []
    image = cv2.imread(filename)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    x_batch = images.reshape(1, image_size, image_size, num_channels)

    sess = tf.Session()

    saver = tf.train.import_meta_graph('ribo-object-classifier.meta')

    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, num_classes))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = sess.run(y_pred, feed_dict=feed_dict_testing)


    inputClass =0

    if className==classes[0]:
        inputClass=0
    elif className == classes[1]:
        inputClass = 1
    elif className == classes[2]:
        inputClass = 2
    elif className == classes[3]:
        inputClass = 3
    elif className == classes[4]:
        inputClass = 4
    elif className == classes[5]:
        inputClass = 5
    elif className == classes[6]:
        inputClass = 6
    elif className == classes[7]:
        inputClass = 7
    elif className == classes[8]:
        inputClass = 8
    elif className == classes[9]:
        inputClass = 9
    elif className == classes[10]:
        inputClass = 10
    elif className == classes[11]:
        inputClass = 11
    m1 = max(result)
    #print(m1)
    m = max(m1)
    j =1;
    for i in m1:
        if i == m:
            break
        j = j + 1

    print('Result: ' + classes[j-1])
    decisionMartrix[inputClass][j-1] = decisionMartrix[inputClass][j-1]+1;
    # print(m)


    # ans= m.argmax(axis=0)
    # print(m)

    # print(i)

print(decisionMartrix)
