import os
import numpy as np
from keras.preprocessing import image

from utils import FLAGS


def get_files(file_dir):
    D= []

    for (dirpath, dirnames, filenames) in os.walk(file_dir):
        for filename in filenames:
            D += [os.path.join(dirpath, filename)]

    temp = np.array([D])
    temp = temp.transpose()

    lists = list(temp[:,0])

    lists.sort()

    image_list = []

    for i in range(len(lists) - FLAGS.time_step):
        image_list.extend(lists[i:i+FLAGS.time_step])

    # print(image_list)
    return image_list


def get_batch_raw(image_list, start):

    img = []

    for i in range(start, start+FLAGS.batch_size):
        image_raw = image.load_img(image_list[i])
        img.append(image_raw)

    return img

def my_generator(image_paths, label, num_samples):

    while True:
        batch_input_list = []
        label_list = np.zeros(shape=[FLAGS.batch_size, FLAGS.road_num])
        # num_samples需要改变
        for i in range(0, num_samples):
            image_list = []
            for j in range(FLAGS.time_step):
                image_raw = image.load_img(image_paths[i+j], grayscale=True)
                image_arr = image.img_to_array(image_raw)
                image_arr = image_arr[:,:,0]
                image_list.append(image_arr)
            img = np.array(image_list)
            batch_input_list.append(img)
            label_list[i % FLAGS.batch_size] = label[i+FLAGS.time_step]

            if len(batch_input_list) == (FLAGS.batch_size):
                # print(np.array(batch_input_list).shape, label_list.shape)
                yield np.array(batch_input_list), label_list

                batch_input_list = []
                label_list = np.zeros(shape=[FLAGS.batch_size, FLAGS.road_num])


def get_batch_tfrecord(dir, image_H, image_W, batch_size, capacity):
    # **1.把所有的 tfrecord 文件名列表写入队列中
    filename_queue = tf.train.string_input_producer([dir],
                                                    shuffle=False)
    # filename_queue = tf.train.string_input_producer(['data/800r_png_training.tfrecord'], num_epochs=1,
    #                                                 shuffle=False)
    # **2.创建一个读取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # **3.根据你写入的格式对应说明读取的格式
    feature = dict()
    feature['image'] = tf.FixedLenFeature([], tf.string)
    for i in range(FLAGS.road_num):
        feature['label_{}'.format(i)] = tf.FixedLenFeature([], tf.int64)
    features = tf.parse_single_example(serialized_example,
                                       features=feature)

    img = features['image']
    # 这里需要对图片进行解码
    img = tf.image.decode_png(img, channels=1)  # 这里，也可以解码为 1 通道
    img = tf.reshape(img, [ image_H,image_W, 1])  # 28*28*3
    img = tf.cast(img, tf.float32)
    print('img3 is', img)

    label = list()
    for i in range(FLAGS.road_num):
        label.append(tf.cast(features['label_{}'.format(i)], tf.int32))

    X_batch,y_batch = tf.train.batch([img,label], batch_size=batch_size, capacity=capacity, num_threads=16)

    X_batch = tf.cast(X_batch, tf.float32)
    y_batch = tf.cast(y_batch, tf.float32)

    y_batch = tf.reshape(y_batch, [-1, FLAGS.time_step, FLAGS.road_num]);

    return X_batch,y_batch

def get_label(start, label):

    label_list = np.zeros(shape=[FLAGS.batch_size, FLAGS.road_num])

    for i in range(FLAGS.batch_size):
        label_list[i] = label[start+i+FLAGS.time_step]

    # label_list = tf.cast(label_list, tf.float32)

    return label_list

if __name__ == '__main__':
    image_list = get_files('E:/test/')
    # get_batch(image_list,998,828,16,64)
    # get_label()



