import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import model


class SummaryBuilder:
    def __init__(self, log_name):
        self.__gap__ = 255 * np.ones([10, 1216, 3], dtype='uint8')

        self.log_folder = self.__make_log_folder__(log_name)

        self.training = None
        self.validation = None
        self.test = None
        if not tf.executing_eagerly():
            self.training = tf.summary.FileWriter(logdir='logs/' + log_name + '_train/')
            self.validation = tf.summary.FileWriter(logdir='logs/' + log_name + '_val/')
            self.test = tf.summary.FileWriter(logdir='logs/' + log_name + '_test/')

    def __make_log_folder__(self, log_name):
        output_folder = 'output/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        log_folder = output_folder + log_name + '/'
        if os.path.exists(log_folder):
            shutil.rmtree(log_folder)

        os.mkdir(log_folder)

        return log_folder

    def build_summary(self, loss, labels, predictions):
        if tf.executing_eagerly():
            self.__get_iou__(predictions, labels)
            return None, None
        else:
            loss_summary = tf.summary.scalar('Loss', loss)
            iou_summary = tf.summary.scalar('IOU', self.__get_iou__(prediction=predictions, truth=labels))

        return loss_summary, iou_summary

    def __make_image__(self, in_image):
        newimage = np.zeros(shape=(in_image.shape[0], in_image.shape[1], 3),
                            dtype='uint8')

        in_image = np.squeeze(in_image, axis=2)

        road_pixels = in_image > 0.0
        newimage[road_pixels] = [255, 0, 255]  # Road Pixels colored
        newimage[np.logical_not(road_pixels)] = [255, 0, 0]  # not road pixels colored

        return newimage

    def __get_iou__(self, prediction, truth):
        truth, prediction = model.mask_out_void(truth, prediction)
        truth = tf.greater(truth, 0, name='RoadTruths')
        prediction = tf.greater(prediction, 0.0, name='RoadPredictions')

        wrong_prediction = tf.cast(tf.logical_xor(prediction, truth, name='WrongPrediction'), dtype=tf.float64)
        correct_prediction = tf.cast(tf.logical_and(prediction, truth, name='CorrectPredictions'), dtype=tf.float64)

        true_positive = tf.reduce_sum(correct_prediction)
        false_positive_plus_negative = tf.reduce_sum(wrong_prediction)

        iou = tf.div(true_positive, tf.add(true_positive, false_positive_plus_negative), name='IOU')

        return iou

    def save_ouput(self, batch_data, ground_truths, segmented_images, image_names, prefix):
        prefix += '/'

        input_image = batch_data[0].astype('uint8')
        ground_truth_image = ground_truths[0]

        segmented_image = segmented_images[0][0]  # Assumption that we only get one image
        sample_name = image_names[0]

        if not os.path.exists(self.log_folder + prefix):
            os.mkdir(self.log_folder + prefix)

        segmented_image = self.__make_image__(segmented_image)
        print('Save: ' + sample_name + ': (' + str(segmented_image.shape) + ')')

        # Stack images in order input image, segmented image and ground truth
        output_image = np.vstack([input_image, self.__gap__, segmented_image, self.__gap__, ground_truth_image])

        save_dpi = 100
        plt.figure(figsize=(output_image.shape[0] / save_dpi, output_image[1] / save_dpi), dpi=save_dpi)
        plt.imshow(output_image)

        plt.savefig(self.log_folder + prefix + sample_name + '.png', bbox_inches='tight', dpi='figure')
        plt.close()


###################
# TEST ONLY
###################
"""
test = np.array([[[[3], [3], [-3]], [[-1], [-3], [3]]]])
test_truth = np.array([[[[1], [0], [0]], [[-1], [1], [1]]]])

tf.enable_eager_execution()
summary = SummaryBuilder('Test')

summary.build_summary(None, test_truth, test)
"""

