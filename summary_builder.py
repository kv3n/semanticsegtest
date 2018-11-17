import cv2
import os
import tensorflow as tf


class SummaryBuilder:
    def __init__(self, log_name):
        self.log_name = log_name

        self.training = tf.summary.FileWriter(logdir='logs/' + log_name + '_train/')
        self.validation = tf.summary.FileWriter(logdir='logs/' + log_name + '_val/')
        self.test = tf.summary.FileWriter(logdir='logs/' + log_name + '_test/')

    def build_summary(self, output):
        output = tf.cast(output, tf.uint8)
        image_summary = tf.map_fn(lambda img: tf.summary.image(name='Output', tensor=tf.expand_dims(img, axis=1)),
                                  elems=output)

        return image_summary

    def save_ouput(self, segmented_images, test_names, show=False):
        num_tests = len(segmented_images)

        for id in range(num_tests):
            if show:

                cv2.imshow('output', segmented_images[id])
                cv2.waitKey()
                cv2.destroyAllWindows()

            cv2.imwrite('output' + self.log_name + '/' + test_names[id])


###################
# TEST ONLY
###################
"""
confusion_matrix = np.array([[2, 4], [3, 1]])
summary = SummaryBuilder('Test', None)
summary.validate_confusion_matrix(confusion_matrix, 0)
"""