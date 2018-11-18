import argparse
import time
import random
from data_feed import *
from model import *
from summary_builder import *

parser = argparse.ArgumentParser(description='Tensorflow Log Name')
parser.add_argument('logname', type=str, nargs='?', help='name of logfile', default='--t')
parser.add_argument('seed', type=int, nargs='?', help='random seed. 0 if true random', default=0)

args = parser.parse_args()
log_name = args.logname
if log_name == '--t':
    log_name = str(time.time())

seed = args.seed
if seed == 0:
    seed = random.randint(0, 1 << 32)
    print('Setting seed: ' + str(seed))

#tf.set_random_seed(seed=seed)

data_feed = Data()

batch_data = tf.placeholder(name='BatchData', dtype=tf.float64,
                            shape=[None, data_feed.image_width, data_feed.image_height, data_feed.image_depth])
true_segmentation = tf.placeholder(name='TrueSegmentation', dtype=tf.int32,
                                   shape=[None, data_feed.image_width, data_feed.image_height, 1])

output, optimize, loss = build_model(image_batch=batch_data, true_segmentation=true_segmentation)

summary_builder = SummaryBuilder(log_name)

with tf.Session() as sess:
    summary_builder.training.add_graph(graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    global_batch_count = 0
    half_epoch_count = 0
    test_epoch_count = 0

    end_of_epochs = False
    while not end_of_epochs:
        try:
            # Run mini-batch
            train_data, train_true_segmentation, _ = data_feed.get_batch_feed(data_type=1)

            _, _ = sess.run([optimize, output], feed_dict={batch_data: train_data,
                                                           true_segmentation: train_true_segmentation})

            run_validation, run_test, end_of_epochs = data_feed.step_train()
            print("Ran Batch" + str(data_feed.global_step))

            if run_validation:
                val_data, val_true_segmentation, val_names = data_feed.get_batch_feed(data_type=2)
                val_output = sess.run([output], feed_dict={batch_data: val_data,
                                                           true_segmentation: val_true_segmentation})

                summary_builder.save_ouput(val_output, val_names, 'val'+str(data_feed.validation_step))
                print("Ran Validation: " + str(data_feed.validation_step))

            if run_test:
                test_data, test_true_segmentation, test_names = data_feed.get_batch_feed(data_type=3)
                test_output = sess.run([output], feed_dict={batch_data: test_data,
                                                            true_segmentation: test_true_segmentation})

                summary_builder.save_ouput(test_output, test_names, 'test'+str(data_feed.test_step))
                print('Ran Test: ' + str(data_feed.test_step))

        except tf.errors.OutOfRangeError:
            print('End of Epochs')
            break
