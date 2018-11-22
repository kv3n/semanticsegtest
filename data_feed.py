import pickle
import numpy as np
import cv2
import os
import random

EPOCHS = 200
TRAIN_SIZE = 199
VAL_SIZE = 45
TEST_SIZE = 45
BATCH_SIZE = 1
VALIDATIONS_PER_EPOCH = 1 / 5
NUM_BATCHES_PER_EPOCH = TRAIN_SIZE // BATCH_SIZE
TOTAL_BATCHES = NUM_BATCHES_PER_EPOCH * EPOCHS
VALIDATION_INTERVAL = NUM_BATCHES_PER_EPOCH // VALIDATIONS_PER_EPOCH
TESTS_PER_EPOCH = 1 / 10
TEST_INTERVAL = int(NUM_BATCHES_PER_EPOCH // TESTS_PER_EPOCH)


class Feed:
    def __init__(self, data, label, names, ground_truth, batch):
        self.data = data
        self.label = label
        self.names = names
        self.batch = batch
        self.ground_truth = ground_truth
        self.feed_size = len(self.data)
        self.current_feed_index = 0

    def get_next_batch(self):
        if self.batch:
            if self.feed_size > 1:
                start_index = random.randint(0, self.feed_size - 1)
            else:
                start_index = 0
            end_index = start_index + 1
        else:
            if self.current_feed_index == self.feed_size:
                self.current_feed_index = 0
                return None, None, None, None

            start_index = self.current_feed_index
            end_index = self.current_feed_index = self.current_feed_index + 1

        return self.data[start_index:end_index], self.names[start_index:end_index], self.label[start_index:end_index], self.ground_truth[start_index:end_index]


class Data:
    def __init__(self):
        def unpickle(file):
            with open('data/label/' + file + '.label', 'rb') as fo:
                label = (pickle.load(fo, encoding='bytes'))
                label = np.expand_dims(label, axis=2)
            return label

        def loadimg(file):
            img = cv2.imread('data/' + file + '.png')
            return img

        def load_files_from_dir(dir_name):
            filenames = os.listdir('data/image/' + dir_name)
            images = []
            labels = []
            gt = []
            final_names = []
            for filename in filenames:
                if not filename == '.DS_Store':
                    stripped_name = filename.rstrip('.png')
                    final_names.append(stripped_name)
                    split = stripped_name.split('_')
                    label_name = split[0] + '_road_' + split[1]
                    gt_name = 'gt/' + dir_name + label_name
                    images.append(loadimg('image/' + dir_name + stripped_name))
                    labels.append(unpickle(dir_name + label_name))
                    gt.append(loadimg(gt_name))

            return images, labels, final_names, gt

        self.global_step = 0
        self.validation_step = 0
        self.test_step = 0

        train_data, train_labels, train_names, train_gt = load_files_from_dir('train/')
        test_data, test_labels, test_names, test_gt = load_files_from_dir('test/')

        self.image_height = train_data[0].shape[0]
        self.image_width = train_data[0].shape[1]
        self.image_depth = train_data[0].shape[2]

        train_end = TRAIN_SIZE
        val_end = train_end + VAL_SIZE
        test_end = TEST_SIZE

        self.train_iterator = self.__make_iterator__(data=train_data,
                                                     names=train_names,
                                                     label=train_labels,
                                                     gt=train_gt,
                                                     start=0,
                                                     end=train_end,
                                                     batch=True)

        self.validation_iterator = self.__make_iterator__(data=train_data,
                                                          names=train_names,
                                                          label=train_labels,
                                                          gt=train_gt,
                                                          start=train_end,
                                                          end=val_end,
                                                          batch=False)

        self.test_iterator = self.__make_iterator__(data=test_data,
                                                    names=test_names,
                                                    label=test_labels,
                                                    gt=test_gt,
                                                    start=0,
                                                    end=test_end,
                                                    batch=False)

    def __make_iterator__(self, data, names, label, gt, start, end, batch=False):
        data = np.array(data)[start:end].astype('float64')
        names = names[start:end]
        label = np.array(label)[start:end].astype('int32')
        gt = np.array(gt)[start:end].astype('uint8')

        return Feed(data=data, label=label, names=names, ground_truth=gt, batch=batch)

    def __get_iterator__(self, data_type):
        if data_type == 1:
            return self.train_iterator
        elif data_type == 2:
            return self.validation_iterator
        else:
            return self.test_iterator

    def get_batch_feed(self, data_type):
        data, names, labels, ground_truth = self.__get_iterator__(data_type).get_next_batch()
        return data, names, labels, ground_truth

    def step_train(self):
        self.global_step += 1
        self.validation_step = self.global_step // VALIDATION_INTERVAL
        self.test_step = self.global_step // TEST_INTERVAL
        run_validation = (self.global_step % VALIDATION_INTERVAL == 0)
        run_test = (self.global_step % TEST_INTERVAL == 0)

        end_test = (self.global_step % TOTAL_BATCHES == 0)

        return run_validation, run_test, end_test
