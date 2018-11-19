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

    def save_ouput(self, segmented_images, image_names, prefix, show=False):
        num_tests = len(segmented_images)

        for id in range(num_tests):
            image = segmented_images[id]
            print('Save: ' + str(image.shape))
            image = np.reshape(image, image.shape[0], image.shape[1])
            print(image.shape)
            plt.figure()
            plt.imshow(image)
            if show:
                plt.show()

            plt.savefig(self.log_folder + prefix + '_' + image_names[id] + '.png')
            plt.close()


###################
# TEST ONLY
###################
"""
confusion_matrix = np.array([[2, 4], [3, 1]])
summary = SummaryBuilder('Test', None)
summary.validate_confusion_matrix(confusion_matrix, 0)
"""