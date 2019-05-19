import tensorflow as tf
import argparse
import time
import matplotlib
# matplotlib.use('agg')
from data_feed import Data
from model import build_model, build_output_functor
import summary_builder

parser = argparse.ArgumentParser(description='Tensorflow Log Name')
parser.add_argument('logname', type=str, nargs='?', help='name of logfile', default='--t')

args = parser.parse_args()
log_name = args.logname
if log_name == '--t':
    log_name = str(time.time())

data_feed = Data()
summary_builder.make_summary_sheet(log_name=log_name)

keras_model = build_model(input_shape=data_feed.data_shape)
keras_output = build_output_functor(keras_model)

"""
loss_summary, iou_summary, iou_calc = summary_builder.summary_sheet.build_summary(loss=loss,
                                                                                  labels=true_segmentation,
                                                                                  predictions=output)

batched_iou_summary = tf.summary.scalar('MEAN_IOU', tensor=batched_iou)

summary_builder.summary_sheet.add_to_training_summary(new_summary=loss_summary)
summary_builder.summary_sheet.add_to_training_summary(new_summary=iou_summary)
"""


def run_batched_testing(data_type, prefix):
    mean_iou = 0.0
    size = 0
    while True:
        data, names, label, gt = data_feed.get_batch_feed(data_type=data_type)
        if data is None:
            break

        results = keras_model.test_on_batch(data, label)
        output = keras_output(data)[0]

        summary_builder.summary_sheet.save_ouput(batch_data=data,
                                                 ground_truths=gt,
                                                 segmented_images=output,
                                                 image_names=names,
                                                 prefix=prefix)

        iou_val = results[1]
        print(prefix + '(' + str(size+1) + ') -> %s' % iou_val)

        mean_iou += iou_val
        size += 1

    mean_iou = mean_iou / size
    print('--------------------------------------------------')

    return mean_iou  # This should be part of the summary


# summary_builder.summary_sheet.training.add_graph(graph=sess.graph)

global_batch_count = 0
half_epoch_count = 0
test_epoch_count = 0

end_of_epochs = False
while not end_of_epochs:
    # Run mini-batch
    train_data, train_name, train_true_segmentation, train_gt = data_feed.get_batch_feed(data_type=1)

    print('Training on {}: {}'.format(train_name[0], train_data.shape))

    results = keras_model.train_on_batch(train_data, train_true_segmentation)

    """
    summary_builder.summary_sheet.show_output(batch_data=train_data,
                                              ground_truths=train_gt,
                                              segmented_images=output,
                                              image_names=train_name)
    """
    # summary_builder.summary_sheet.training.add_summary(summaries, global_step=data_feed.global_step)

    run_validation, run_test, end_of_epochs = data_feed.step_train()
    print('Ran Batch: ' + str(data_feed.global_step))
    print('------------------------------------------------------------------')

    if run_validation:
        val_iou_val = run_batched_testing(data_type=2, prefix='test'+str(int(data_feed.test_step)))
        # summary_builder.summary_sheet.validation.add_summary(val_iou_summary, data_feed.validation_step)

        print('Ran Validation: {} with {} mean acc'.format(data_feed.validation_step, val_iou_val))

        keras_model.save_weights('weight_{}.hd5'.format(data_feed.validation_step))

    if run_test:
        test_iou_val = run_batched_testing(data_type=3, prefix='test'+str(int(data_feed.test_step)))
        # summary_builder.summary_sheet.test.add_summary(test_iou_summary, data_feed.test_step)

        print('Ran Test: {} with {} mean acc'.format(data_feed.test_step, test_iou_val))
