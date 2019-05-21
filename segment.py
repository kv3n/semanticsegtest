import tensorflow as tf
import argparse
import time
import matplotlib
matplotlib.use('agg')
from data_feed import Data
from model import build_model, build_output_functor
from summary_builder import SummaryBuilder

parser = argparse.ArgumentParser(description='Tensorflow Log Name')
parser.add_argument('logname', type=str, nargs='?', help='name of logfile', default='--t')

args = parser.parse_args()
log_name = args.logname
if log_name == '--t':
    log_name = str(time.time())

data_feed = Data()
summary_sheet = SummaryBuilder(log_name=log_name)

keras_model = build_model(input_shape=data_feed.data_shape)
keras_output = build_output_functor(keras_model)

summary_sheet.build_histogram(keras_model)


def run_batched_testing(data_type, prefix):
    mean_iou = 0.0
    mean_loss = 0.0
    mean_acc = 0.0
    size = 0
    while True:
        data, names, label, gt = data_feed.get_batch_feed(data_type=data_type)
        if data is None:
            break

        results = keras_model.test_on_batch(data, label)
        output = keras_output(data)[0]

        summary_sheet.save_ouput(batch_data=data,
                                 ground_truths=gt,
                                 segmented_images=output,
                                 image_names=names,
                                 prefix=prefix)

        iou_val = summary_sheet.get_iou(prediction=output, truth=label)
        print(prefix + '(' + str(size+1) + ') -> %s' % iou_val)

        mean_iou += iou_val
        mean_loss += results[0]
        mean_acc += results[1]
        size += 1

    mean_iou = mean_iou / size
    mean_acc = mean_acc / size
    mean_loss = mean_loss / size
    print('--------------------------------------------------')

    return [mean_loss, mean_acc, mean_iou]


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
    output = keras_output(train_data)[0]
    results += [summary_sheet.get_iou(prediction=output, truth=train_true_segmentation)]
    with summary_sheet.training.as_default():
        summary_sheet.commit_sheet(results, step=data_feed.global_step)

    """
    summary_sheet.show_output(batch_data=train_data,
                              ground_truths=train_gt,
                              segmented_images=output,
                              image_names=train_name)
    """

    run_validation, run_test, end_of_epochs = data_feed.step_train()
    print('Ran Batch: ' + str(data_feed.global_step))
    print('------------------------------------------------------------------')

    if run_validation:
        results = run_batched_testing(data_type=2, prefix='test'+str(int(data_feed.validation_step)))
        with summary_sheet.validation.as_default():
            summary_sheet.commit_sheet(results, step=data_feed.validation_step)

        print('Ran Validation: {} with {} mean acc'.format(data_feed.validation_step, results[1]))

        keras_model.save_weights('weight_{}.hd5'.format(data_feed.validation_step))

    if run_test:
        results = run_batched_testing(data_type=3, prefix='test'+str(int(data_feed.test_step)))
        with summary_sheet.test.as_default():
            summary_sheet.commit_sheet(results, step=data_feed.test_step)

        print('Ran Test: {} with {} mean acc'.format(data_feed.test_step, results[1]))
