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
                image_raw = image.load_img(image_paths[i*FLAGS.time_step+j], grayscale=True)
                # print(image_paths[i*FLAGS.time_step+j])
                image_arr = image.img_to_array(image_raw)
                image_list.append(image_arr)
            img = np.concatenate(image_list, axis=-1)
            batch_input_list.append(img)
            label_list[i % FLAGS.batch_size] = label[i+FLAGS.time_step]

            if len(batch_input_list) == (FLAGS.batch_size):
                # print(np.array(batch_input_list).shape, label_list.shape)
                yield np.array(batch_input_list), label_list

                batch_input_list = []
                label_list = np.zeros(shape=[FLAGS.batch_size, FLAGS.road_num])



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



