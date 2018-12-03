import tensorflow as tf
from data_process import load_data_files,padding_sentences,batch_iter
from word2vec import embedding_sentences
import time,os
import numpy as np
from inference import TextCnn
import datetime
# data loading params
tf.app.flags.DEFINE_integer('val_sample_percentage',0.1,'peecentage of training data to use for testing')
tf.app.flags.DEFINE_integer('test_sample_percentage',0.1,'peecentage of training data to use for validation')
tf.app.flags.DEFINE_string('training_data','./data/cnn_data.csv','training data filename')
tf.app.flags.DEFINE_string('word2vec_model', 'word2vec_model', 'embedding word2vec of chinese word')
tf.app.flags.DEFINE_string('trained_word2vec_model', 'word2vec_model', 'embedding word2vec of chinese word')
tf.flags.DEFINE_integer("num_labels", 2, "Number of labels for data. (default: 2)")
tf.flags.DEFINE_string("model_save_path", 'model', "model save)")


# Model Hparams
tf.flags.DEFINE_integer('embedding_dim', 128, 'dimensionality of character')
tf.flags.DEFINE_string('filter_size','3,4,5','filter size')
tf.flags.DEFINE_integer('number_filter',128,'number of filter')
tf.app.flags.DEFINE_float('dropout',0.7,'dropout')
tf.app.flags.DEFINE_float('L2_reg_lambda',0.001,'L2')
tf.app.flags.DEFINE_float('learning_rate',0.01,'learning rate')
tf.app.flags.DEFINE_float('moving_averages_op',0.99,'moving averages operation')
tf.app.flags.DEFINE_float('learning_rate_decay',0.99,'learning rate decay')



#Model params
tf.app.flags.DEFINE_integer('epochs',20,'number of epochs')
tf.app.flags.DEFINE_integer('batch_size',256,'size of batch')
tf.flags.DEFINE_integer('evaluate_every', 500, 'evaluate_every')
tf.flags.DEFINE_integer('checkpoint_every', 1000, 'saving...')
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("padding_sentence_length", 15, "padding sentence length")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Parse parameters from commands
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print('\nParameters:')
for attr, val in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), val))

x_text ,y = load_data_files(FLAGS.training_data)

# Prepare output directory for models and summaries
timestamp = str(int(time.time()))
out_dir =os.path.dirname(os.path.join(os.path.curdir,'data/runs',timestamp))
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# Get embedding vector
sentences, max_document_length = padding_sentences(x_text, '<PADDING>',FLAGS.padding_sentence_length)

x =np.array(embedding_sentences(sentences=sentences,file_to_save=os.path.join(out_dir,FLAGS.word2vec_model),file_to_load=os.path.join(out_dir,FLAGS.word2vec_model)))

# Shuffle data randomly
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/valid/test set
test_sample_index = -1 * int(FLAGS.test_sample_percentage * float(len(y)))

x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]
val_sample_index = -1 * int(FLAGS.val_sample_percentage * float(len(y_train)))
x_val, y_val = x_train[val_sample_index:], y_train[val_sample_index:]
print("Train/Test/val split: {:d}/{:d}/{:d}".format(len(y_train), len(y_test),len(y_val)))

def train():
    input_x = tf.placeholder(tf.float32, [None, x_train.shape[1], x_train.shape[2]], name='input_x')
    input_y = tf.placeholder(tf.float32, [None, y_train.shape[1]], name='input_y')
    regularizer =tf.contrib.layers.l2_regularizer(FLAGS.L2_reg_lambda)

    cnn = TextCnn(input_x=input_x,sequence_length=x_train.shape[1],
                     num_classes=y_train.shape[1],
                     embedding_size = x_train.shape[2],
                     filter_sizes=list(map(int,FLAGS.filter_size.split(','))),
                     num_filters =FLAGS.number_filter,
                     regularizer=regularizer)
    y= cnn.socres
    global_step = tf.Variable(0,trainable=False)
    variable_averages= tf.train.ExponentialMovingAverage(FLAGS.moving_averages_op,global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # with tf.name_scope('loss'):
    loss  = tf.nn.softmax_cross_entropy_with_logits(logits=y ,labels=input_y)
    losses = tf.reduce_mean(loss) +tf.add_n(tf.get_collection('losses'))

    learning_rate=tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step,
        len(y_train)/FLAGS.batch_size,
        FLAGS.learning_rate_decay,
        staircase=True
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(losses,global_step=global_step)
    # grads_and_vars = optimizer.compute_gradients(losses)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(cnn.predictions ,tf.argmax(input_y ,1))
        accuracy1 = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')


    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        batchs = batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.epochs)



        def val_test_step(x_batch, y_batch,model):
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, loss, accuracy = sess.run(
                [global_step, losses,accuracy1], feed_dict=feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            if model == 'test':
                print('test  {}: step:{} , loss:{} , acc:{}'.format(time_str, step, loss, accuracy))
            else:
                print('val  {}: step:{} , loss:{} , acc:{}'.format(time_str, step, loss, accuracy))


        def train_st(x_batch, y_batch):
            feed_dict = {
                input_x: x_batch,
                input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout
            }
            _, step,  loss_value, accuracy = sess.run(
                [train_op, global_step, losses,accuracy1], feed_dict=feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print('{}:step:{} , loss:{} , acc:{}'.format(time_str, step, loss_value, accuracy))
            if step % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                print('\n evaluate_every')
                val_test_step(x_test, y_test, 'test')
                val_test_step(x_val, y_val, 'val')
                saver.save(sess, os.path.join(FLAGS.model_save_path, 'model.ckpt'), global_step=step)

        for batch in batchs:
            x_batch, y_batch = zip(*batch)
            train_st(x_batch,y_batch)



if __name__ == '__main__':

    train()

