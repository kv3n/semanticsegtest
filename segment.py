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

tf.set_random_seed(seed=seed)

data_type = tf.placeholder(name='DataType', dtype=tf.uint8)
data_feed = Data(datatype_placeholder=data_type)
batch_data, batch_names, true_segmentation = data_feed.get_batch_feed()

output, optimize, loss = build_model(image_batch=batch_data, true_segmentation=true_segmentation)

summary_builder = SummaryBuilder(log_name)

with tf.Session() as sess:
    summary_builder.training.add_graph(graph=sess.graph)

    sess.run(tf.global_variables_initializer())

    global_batch_count = 0
    half_epoch_count = 0
    test_epoch_count = 0

    while True:
        try:
            # Run mini-batch
            ex = sess.run([batch_data], feed_dict={data_type: 1})

            _, _ = sess.run([loss, output], feed_dict={data_type: 1})

            run_validation, run_test = data_feed.step_train()
            print("Ran Batch" + str(data_feed.global_step))

            if run_validation:
                val_output = sess.run([output], feed_dict={data_type: 2})

                summary_builder.save_ouput(val_output, batch_names, 'val'+str(data_feed.validation_step), show=True)
                print("Ran Validation: " + str(data_feed.validation_step))

            if run_test:
                test_output = sess.run([output], feed_dict={data_type: 3})

                summary_builder.save_ouput(test_output, batch_names, 'test'+str(data_feed.test_step), show=True)
                print('Ran Test: ' + str(data_feed.test_step))

        except tf.errors.OutOfRangeError:
            print('End of Epochs')
            break
