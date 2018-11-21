import tensorflow as tf
"""
###################################
# TESTING ONLY ####################
###################################
tf.enable_eager_execution()
"""

import argparse
import time
import matplotlib
matplotlib.use('agg')
from data_feed import *
import model
import summary_builder
import seed_gen

parser = argparse.ArgumentParser(description='Tensorflow Log Name')
parser.add_argument('logname', type=str, nargs='?', help='name of logfile', default='--t')
parser.add_argument('seed', type=int, nargs='?', help='random seed. 0 if true random', default=0)

args = parser.parse_args()
log_name = args.logname
if log_name == '--t':
    log_name = str(time.time())

seed_gen.make_seed_distributor(seed=args.seed)
print('Using Seed: ' + str(seed_gen.seed_distributor.random_seed))

data_feed = Data()
summary_builder.make_summary_sheet(log_name=log_name)

if tf.executing_eagerly():
    batch_data, train_name, true_segmentation, train_gt = data_feed.get_batch_feed(data_type=1)
else:
    batch_data = tf.placeholder(name='BatchData', dtype=tf.float64,
                                shape=[None, data_feed.image_height, data_feed.image_width, data_feed.image_depth])
    true_segmentation = tf.placeholder(name='TrueSegmentation', dtype=tf.int32,
                                       shape=[None, data_feed.image_height, data_feed.image_width, 1])

output, optimize, loss = model.build_model(image_batch=batch_data, true_segmentation=true_segmentation)

loss_summary, iou_summary = summary_builder.summary_sheet.build_summary(loss=loss, labels=true_segmentation, predictions=output)
summary_builder.summary_sheet.add_to_training_summary(new_summary=loss_summary)
summary_builder.summary_sheet.add_to_training_summary(new_summary=iou_summary)


def run_batched_testing(sess, data_type, prefix):
    mean_iou = 0.0
    size = 0
    while True:
        data, names, true_segmentation, gt = data_feed.get_batch_feed(data_type=data_type)
        if data is None:
            break

        output_val, iou_val = sess.run([output, iou_summary], feed_dict={batch_data: data,
                                                                         true_segmentation: true_segmentation})

        summary_builder.summary_sheet.save_ouput(batch_data=data,
                                                 ground_truths=gt,
                                                 segmented_images=output_val,
                                                 image_names=names,
                                                 prefix=prefix + str(data_feed.validation_step))

        print(prefix + '(' + str(size+1) + ') -> ' + str(iou_val))

        mean_iou += iou_val
        size += 1

    mean_iou = mean_iou / size
    print('--------------------------------------------------')

    return tf.Summary().value.add(tag='mean_iou', simple_value=mean_iou)


with tf.Session() as sess:
    summary_builder.summary_sheet.training.add_graph(graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    global_batch_count = 0
    half_epoch_count = 0
    test_epoch_count = 0

    end_of_epochs = False
    while not end_of_epochs:
        try:
            # Run mini-batch
            train_data, train_name, train_true_segmentation, train_gt = data_feed.get_batch_feed(data_type=1)

            print(str(train_name[0]) + ': ' + str(train_data.shape))

            _, output_results, summaries = sess.run([optimize, output, summary_builder.summary_sheet.training_summaries],
                                                    feed_dict={batch_data: train_data,
                                                               true_segmentation: train_true_segmentation})

            summary_builder.summary_sheet.training.add_summary(summaries, global_step=data_feed.global_step)

            run_validation, run_test, end_of_epochs = data_feed.step_train()
            print('Ran Batch: ' + str(data_feed.global_step))
            print('Max: ' + str(np.amax(output_results)))
            print('Min: ' + str(np.amin(output_results)))

            print('------------------------------------------------------------------')

            if run_validation:
                val_iou_summary = run_batched_testing(sess=sess, data_type=2, prefix='val')
                summary_builder.summary_sheet.validation.add_summary(val_iou_summary, data_feed.validation_step)

                print("Ran Validation: " + str(data_feed.validation_step))

            if run_test:
                test_iou_summary = run_batched_testing(sess=sess, data_type=3, prefix='test')
                summary_builder.summary_sheet.test.add_summary(test_iou_summary, data_feed.test_step)

                print("Ran Test: " + str(data_feed.test_step))

        except tf.errors.OutOfRangeError:
            print('End of Epochs')
            break
