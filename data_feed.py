import pickle
import numpy as np
import tensorflow as tf
import cv2
import os

tf.enable_eager_execution()

EPOCHS = 50
TRAIN_SIZE = 199
VAL_SIZE = 45
TEST_SIZE = 45
BATCH_SIZE = 64
VALIDATIONS_PER_EPOCH = 2
NUM_BATCHES_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
VALIDATION_INTERVAL = NUM_BATCHES_PER_EPOCH // VALIDATIONS_PER_EPOCH
TESTS_PER_EPOCH = 0.1
TEST_INTERVAL = int(NUM_BATCHES_PER_EPOCH // TESTS_PER_EPOCH)


class Data:
    def __init__(self):
        def unpickle(file):
            with open('data/label/' + file + '.label', 'rb') as fo:
                label = pickle.load(fo, encoding='bytes')
            return label

        def loadimg(file):
            img = cv2.imread('data/image/' + file + '.png')
            return img

        def load_files_from_dir(dir_name):
            filenames = os.listdir('data/image/' + dir_name)
            images = []
            labels = []
            for filename in filenames:
                if not filename == '.DS_Store':
                    stripped_name = filename.rstrip('.png')
                    split = stripped_name.split('_')
                    label_name = split[0] + '_road_' + split[1]
                    images.append(loadimg(dir_name + stripped_name))
                    labels.append(unpickle(dir_name + label_name))

            return images, labels

        train_data, train_labels = load_files_from_dir('train/')
        test_data, test_labels = load_files_from_dir('test/')

        print('loaded data')


"""
        self.train_iterator, self.train_len = self.__make_iterator__(raw=train_raw,
                                                                     start=0,
                                                                     end=train_end,
                                                                     epochs=EPOCHS,
                                                                     batch_size=BATCH_SIZE)

        self.validation_iterator, self.validation_len = self.__make_iterator__(raw=train_raw,
                                                                               start=train_end,
                                                                               end=val_end,
                                                                               epochs=int(EPOCHS * VALIDATIONS_PER_EPOCH))

        self.test_iterator, self.test_len = self.__make_iterator__(raw=test_raw,
                                                                   start=0,
                                                                   end=test_end,
                                                                   epochs=int(EPOCHS * TESTS_PER_EPOCH))

        self.mean_image = tf.Variable(initial_value=self.__mean_image_initializer__(np.array(train_raw[b'data'])[0:train_end]),
                                      name='MeanImage',
                                      trainable=False)

        self.mapping = dict(set(zip(train_raw[b'fine_labels'], train_raw[b'coarse_labels'])))

    def __make_iterator__(self, raw, start, end, epochs, batch_size=-1):
        epochs = max(1, epochs)
        if batch_size < 0:
            batch_size = (end - start)

        data = np.array(raw[b'data'])[start:end]
        coarse_labels = np.array(raw[b'coarse_labels'])[start:end].astype('int32')
        fine_labels = np.array(raw[b'fine_labels'])[start:end].astype('int32')

        dataset = tf.data.Dataset.from_tensor_slices((data, fine_labels, coarse_labels))
        dataset = dataset.shuffle(buffer_size=(end-start), reshuffle_each_iteration=True) \
                         .repeat(count=epochs) \
                         .batch(batch_size=batch_size)
        return dataset.make_one_shot_iterator(), len(data)

    def __get_train_iterator__(self):
        return self.train_iterator.get_next(name='TrainIterator')

    def __get_validation_iterator__(self):
        return self.validation_iterator.get_next(name='ValidationIterator')

    def __get_test_iterator__(self):
        return self.test_iterator.get_next(name='TestIterator')

    def __mean_image_initializer__(self, train_data):
        cast_data = tf.cast(train_data, tf.float32)
        normalized_data = tf.divide(cast_data, tf.constant(255.0, tf.float32))
        return tf.reduce_mean(normalized_data,
                              axis=0,
                              keepdims=True)

    def __get_iterator__(self):
        return tf.case(pred_fn_pairs={self.is_train(): self.__get_train_iterator__,
                                      self.is_validation(): self.__get_validation_iterator__,
                                      self.is_test(): self.__get_test_iterator__},
                       exclusive=True,
                       name='DataSelector')

    def get_batch_feed(self):
        input_data, fine_label, coarse_label = self.__get_iterator__()
        input_mean_shift = tf.subtract(x=tf.divide(tf.cast(input_data, tf.float32), 255.0),
                                       y=self.mean_image,
                                       name='MeanShift')
        return tf.transpose(a=tf.reshape(tensor=input_mean_shift,
                                         shape=[-1, 3, 32, 32]),
                            perm=[0, 2, 3, 1],
                            name='MakeImage'), input_data, fine_label, coarse_label

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
    """

###################
# TEST ONLY
###################
data = Data()

"""
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