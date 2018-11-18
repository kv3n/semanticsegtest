import pickle
import numpy as np
import tensorflow as tf
import cv2
import os

EPOCHS = 50
TRAIN_SIZE = 199
VAL_SIZE = 45
TEST_SIZE = 45
BATCH_SIZE = 1
VALIDATIONS_PER_EPOCH = 2
NUM_BATCHES_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VALIDATION_INTERVAL = NUM_BATCHES_PER_EPOCH // VALIDATIONS_PER_EPOCH
TESTS_PER_EPOCH = 1.0 / EPOCHS
TEST_INTERVAL = int(NUM_BATCHES_PER_EPOCH // TESTS_PER_EPOCH)


class Data:
    def __init__(self, datatype_placeholder):
        def unpickle(file):
            with open('data/label/' + file + '.label', 'rb') as fo:
                label = (pickle.load(fo, encoding='bytes'))
                label = np.expand_dims(label, axis=2)
                label += 1
            return label

        def loadimg(file):
            img = cv2.imread('data/image/' + file + '.png')
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

        self.data_type = datatype_placeholder

        self.global_step = 0
        self.validation_step = 0
        self.test_step = 0

        train_data, train_labels, train_names = load_files_from_dir('train/')
        test_data, test_labels, test_names = load_files_from_dir('test/')

        train_end = TRAIN_SIZE
        val_end = train_end + VAL_SIZE
        test_end = TEST_SIZE

        self.train_iterator, self.train_len = self.__make_iterator__(data=train_data,
                                                                     names=train_names,
                                                                     label=train_labels,
                                                                     start=0,
                                                                     end=train_end,
                                                                     epochs=EPOCHS,
                                                                     batch_size=BATCH_SIZE)

        self.validation_iterator, self.validation_len = self.__make_iterator__(data=train_data,
                                                                               names=train_names,
                                                                               label=train_labels,
                                                                               start=train_end,
                                                                               end=val_end,
                                                                               epochs=int(EPOCHS * VALIDATIONS_PER_EPOCH))

        self.test_iterator, self.test_len = self.__make_iterator__(data=test_data,
                                                                   names=test_names,
                                                                   label=test_labels,
                                                                   start=0,
                                                                   end=test_end,
                                                                   epochs=int(EPOCHS * TESTS_PER_EPOCH))

    def __make_iterator__(self, data, names, label, start, end, epochs, batch_size=-1):
        epochs = max(1, epochs)

        data = np.array(data)[start:end].astype('float64')
        names = names[start:end]
        label = np.array(label)[start:end].astype('int32')

        dataset = tf.data.Dataset.from_tensor_slices((data, names, label))
        dataset = dataset.repeat(count=epochs)

        if batch_size <= 0:
            batch_size = end - start

        dataset = dataset.batch(batch_size=batch_size)

        return dataset.make_one_shot_iterator(), len(data)

    def __get_train_iterator__(self):
        return self.train_iterator.get_next(name='TrainIterator')

    def __get_validation_iterator__(self):
        return self.validation_iterator.get_next(name='ValidationIterator')

    def __get_test_iterator__(self):
        return self.test_iterator.get_next(name='TestIterator')

    def __get_iterator__(self):
        return tf.case(pred_fn_pairs={self.is_train(): self.__get_train_iterator__,
                                      self.is_validation(): self.__get_validation_iterator__,
                                      self.is_test(): self.__get_test_iterator__},
                       exclusive=True,
                       name='DataSelector')

    def get_batch_feed(self):
        data, names, labels = self.__get_iterator__()
        return data, names, labels

    def step_train(self):
        self.global_step += 1
        self.validation_step = self.global_step // VALIDATION_INTERVAL
        self.test_step = self.global_step // TEST_INTERVAL
        run_validation = (self.global_step % VALIDATION_INTERVAL == 0)
        run_test = (self.global_step % TEST_INTERVAL == 0)

        return run_validation, run_test

    def is_train(self):
        return tf.equal(self.data_type, 1)

    def is_validation(self):
        return tf.equal(self.data_type, 2)

    def is_test(self):
        return tf.equal(self.data_type, 3)


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