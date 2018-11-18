import cv2
import os
import shutil
import tensorflow as tf


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

    def save_ouput(self, segmented_images, image_names, prefix, show=False):
        segmented_images = segmented_images
        num_tests = len(segmented_images)

        for id in range(num_tests):
            if show:
                cv2.imshow('output', segmented_images[id])
                cv2.waitKey()
                cv2.destroyAllWindows()

            cv2.imwrite(self.log_folder + prefix + '_' + image_names[id] + '.png', segmented_images[id])


###################
# TEST ONLY
###################
"""
confusion_matrix = np.array([[2, 4], [3, 1]])
summary = SummaryBuilder('Test', None)
summary.validate_confusion_matrix(confusion_matrix, 0)
"""