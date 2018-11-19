import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class SummaryBuilder:
    def __init__(self, log_name):
        self.log_folder = self.__make_log_folder__(log_name)

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
        loss_summary = tf.summary.scalar('Loss', loss)

        return loss_summary

    def __make_image__(self, in_image):
        newimage = np.zeros(shape=(in_image.shape[0], in_image.shape[1], 3),
                            dtype='uint8')

        in_image = np.squeeze(in_image, axis=2)

        road_pixels = in_image > 0.0
        newimage[road_pixels] = [255, 0, 255]  # Road Pixels colored
        newimage[np.logical_not(road_pixels)] = [255, 0, 0]  # not road pixels colored

        return newimage

    def save_ouput(self, segmented_images, image_names, prefix, show=False):
        segmented_images = segmented_images[0]
        num_tests = len(segmented_images)

        for id in range(num_tests):
            newimage = self.__make_image__(segmented_images[id])

            print('Save: ' + image_names[id] + ': (' + str(newimage.shape) + ')')
            plt.figure()
            plt.imshow(newimage)
            if show:
                plt.show()

            plt.savefig(self.log_folder + prefix + '_' + image_names[id] + '.png')
            plt.close()


###################
# TEST ONLY
###################
"""
test = [np.array([[[[1], [-1]], [[-1], [1]]]])]
summary = SummaryBuilder('Test')
summary.save_ouput(test, 'blah', '2', show=False)
"""
