import pickle
import numpy as np
import matplotlib.image as mpimg
import os
import random

EPOCHS = 50
TRAIN_SIZE = 199
VAL_SIZE = 45
TEST_SIZE = 45
BATCH_SIZE = 1
VALIDATIONS_PER_EPOCH = 20
NUM_BATCHES_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
TOTAL_BATCHES = NUM_BATCHES_PER_EPOCH * EPOCHS
VALIDATION_INTERVAL = NUM_BATCHES_PER_EPOCH // VALIDATIONS_PER_EPOCH
TESTS_PER_EPOCH = 0.1
TEST_INTERVAL = int(NUM_BATCHES_PER_EPOCH // TESTS_PER_EPOCH)


class Feed:
    def __init__(self, data, label, names, batch):
        self.data = data
        self.label = label
        self.names = names
        self.batch = batch
        self.feed_size = len(self.data)

    def get_next_batch(self):
        if self.batch:
            start_index = random.randint(0, self.feed_size)
            end_index = start_index + 1
        else:
            start_index = 0
            end_index = self.feed_size

        return self.data[start_index:end_index], self.label[start_index:end_index], self.names[start_index:end_index]


class Data:
    def __init__(self):
        def unpickle(file):
            with open('data/label/' + file + '.label', 'rb') as fo:
                label = (pickle.load(fo, encoding='bytes'))
                label = np.expand_dims(label, axis=2)
                label += 1
            return label

        def loadimg(file):
            img = mpimg.imread('data/image/' + file + '.png')
            return img

        def load_files_from_dir(dir_name):
            filenames = os.listdir('data/image/' + dir_name)
            images = []
            labels = []
            final_names = []
            for filename in filenames:
                if not filename == '.DS_Store':
                    stripped_name = filename.rstrip('.png')
                    final_names.append(stripped_name)
                    split = stripped_name.split('_')
                    label_name = split[0] + '_road_' + split[1]
                    images.append(loadimg(dir_name + stripped_name))
                    labels.append(unpickle(dir_name + label_name))

            return images, labels, final_names

        self.global_step = 0
        self.validation_step = 0
        self.test_step = 0

        train_data, train_labels, train_names = load_files_from_dir('train/')
        test_data, test_labels, test_names = load_files_from_dir('test/')

        self.image_width = train_data[0].shape[0]
        self.image_height = train_data[0].shape[1]
        self.image_depth = train_data[0].shape[2]

        train_end = TRAIN_SIZE
        val_end = train_end + VAL_SIZE
        test_end = TEST_SIZE

        self.train_iterator = self.__make_iterator__(data=train_data,
                                                     names=train_names,
                                                     label=train_labels,
                                                     start=0,
                                                     end=train_end,
                                                     batch=True)

        self.validation_iterator = self.__make_iterator__(data=train_data,
                                                          names=train_names,
                                                          label=train_labels,
                                                          start=train_end,
                                                          end=val_end,
                                                          batch=True)

        self.test_iterator = self.__make_iterator__(data=test_data,
                                                    names=test_names,
                                                    label=test_labels,
                                                    start=0,
                                                    end=test_end,
                                                    batch=True)

    def __make_iterator__(self, data, names, label, start, end, batch=False):
        data = np.array(data)[start:end].astype('float64')
        names = names[start:end]
        label = np.array(label)[start:end].astype('int32')

        return Feed(data, label, names, batch)

    def __get_iterator__(self, data_type):
        if data_type == 1:
            return self.train_iterator
        elif data_type == 2:
            return self.validation_iterator
        else:
            return self.test_iterator

    def get_batch_feed(self, data_type):
        data, names, labels = self.__get_iterator__(data_type).get_next_batch()
        return data, names, labels

    def step_train(self):
        self.global_step += 1
        self.validation_step = self.global_step // VALIDATION_INTERVAL
        self.test_step = self.global_step // TEST_INTERVAL
        run_validation = (self.global_step % VALIDATION_INTERVAL == 0)
        run_test = (self.global_step % TEST_INTERVAL == 0)

        end_test = (self.global_step % TOTAL_BATCHES == 0)

        return run_validation, run_test, end_test


###################
# TEST ONLY
###################
"""
data = Data()

data_type = tf.placeholder(name='DataType', dtype=tf.uint8)
data = Data(data_type)
input_layer, fine_label, coarse_label = data.get_batch_feed()

with tf.Session() as data_sess:
    data_sess.run(tf.global_variables_initializer())
    print(data.mean_image.eval())

    while True:
        try:
            d = data_sess.run([fine_label], feed_dict={data_type: 3})
        except tf.errors.OutOfRangeError:
            print('End of Epochs')
            break
"""

##################
# DEBUG HELPERS
##################

"""
def debug_draw_batch(batch, size):
    vstack = np.zeros([1, 32 * size + 1, 3], 'uint8')

    for r in range(size):
        hstack = np.zeros([32, 1, 3], 'uint8')
        for c in range(size):
            hstack = np.hstack((hstack, batch[0][r * size + c]))
        vstack = np.vstack((vstack, hstack))

    plt.figure()
    plt.imshow(vstack)
    plt.show()
"""