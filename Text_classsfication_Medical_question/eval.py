import time
import tensorflow as tf
from train import x_test,y_test,x_val,y_val
from inference import TextCnn
from train import FLAGS
def evaluate():
    input_x = tf.placeholder(tf.float32, [None, x_test.shape[1], x_test.shape[2]], name='input_x')
    input_y = tf.placeholder(tf.float32, [None, y_test.shape[1]], name='input_y')
    cnn = TextCnn(input_x=input_x,sequence_length=x_test.shape[1],
                     num_classes=y_test.shape[1],
                     embedding_size = x_test.shape[2],
                     filter_sizes=list(map(int,FLAGS.filter_size.split(','))),
                     num_filters =FLAGS.number_filter,
                     regularizer=None)
    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(cnn.predictions, tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_averages_op)
    variable_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variable_to_restore)
    while True:
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                FLAGS.model_save_path
            )
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                accuracy_score_test = sess.run(accuracy,feed_dict={input_x:x_test,input_y:y_test,cnn.dropout_keep_prob:1.0})
                accuracy_score_val = sess.run(accuracy,feed_dict={input_x:x_val,input_y:y_val,cnn.dropout_keep_prob:1.0})
                print("accuracy of validation {%s}；accuracy of testing {%s}"%(accuracy_score_val,accuracy_score_test))
            else:
                print("NO checkpoint file found")
                return
        time.sleep(60)


if __name__ == '__main__':
    evaluate()
