import tensorflow as tf




def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights



class TextCnn:
    def __init__(self,input_x,sequence_length,num_classes,embedding_size,filter_sizes,num_filters,regularizer):
        # Placeholders for input, output, dropout
        self.input_x= input_x
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        with tf.device('/cpu:0'),tf.name_scope('embedding'):
            self.embedding_chars = self.input_x
            self.embedding_chars_expanded=tf.expand_dims(self.embedding_chars,-1)  # shape : [ batch , height , width , 1 ]
        pooled_outputs = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.variable_scope('conv_maxpool-%s' % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                weights = get_weight_variable(filter_shape,regularizer)
                # W1 = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name='W1')
                biases = tf.get_variable('biases',[num_filters],initializer=tf.constant_initializer(0.0))
                # b1 = tf.Variable(tf.constant(0.1,shape=[num_filters]),name='b1')
                conv = tf.nn.conv2d(self.embedding_chars_expanded,weights,strides=[1,1,1,1],padding='VALID',name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv,biases),name='relu') #[ batch , sequence_length - filter_sizes + 1 , 1 , num_filters ]
                pool = tf.nn.max_pool(h,ksize=[1,sequence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID',name='pool')
                pooled_outputs.append(pool)

        num_filters_size = num_filters*len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool,shape=[-1,num_filters_size])
        with tf.variable_scope('dropout'):
            self.dropout = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)
        with tf.variable_scope('fully_connected'):
            weights = get_weight_variable([num_filters_size,100], regularizer)
            # W2 =tf.Variable(tf.truncated_normal([num_filters_size,100],stddev=0.1),name='W2')
            biases = tf.get_variable('biases', [100], initializer=tf.constant_initializer(0.0))
            # b2 = tf.Variable(tf.constant(0.1,shape=[100,],name='b2'))
            self.connected = tf.nn.xw_plus_b(self.dropout,weights,biases)
        with tf.variable_scope('output'):
            weights = get_weight_variable([100, num_classes], regularizer)
            # W3 = tf.Variable(tf.truncated_normal([100, num_classes],stddev=0.1),name='W3')
            biases = tf.get_variable('biases', [num_classes], initializer=tf.constant_initializer(0.0))
            # b3 = tf.Variable(tf.constant(0.1, shape=[num_classes], name='b'))
            self.socres = tf.nn.xw_plus_b(self.connected, weights, biases,name='socres')
            self.predictions = tf.argmax(self.socres, 1, name='predictions')
