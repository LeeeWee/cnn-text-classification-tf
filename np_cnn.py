import tensorflow as tf

class NP_CNN_Rand(object):
    """
    A CNN which leverages both lexical and program structure information to learn unified features
    from natural language and source code in programming language for automatically locating the
    potential buggy source code according to bug report.
    """
    def __init__(
            self, n_vocab_size, n_sequence_length, n_filter_sizes, n_num_filters,
            p_vocab_size, p_statement_length, p_statement_filter_size, p_statement_filters,
            p_semantics_filter_sizes, p_semantics_filters, embedding_size,
            l2_reg_lambda=0.0, add_class_weights=False, class_weights=None):

        # Placeholders for input, output and dropout
        self.input_n = tf.placeholder(tf.int32, [None, n_sequence_length], name="input_n")
        self.input_p = tf.placeholder(tf.int32, [None, None, p_statement_length], name="input_p")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("n_embedding"):
            self.n_W = tf.Variable(
                tf.random_uniform([n_vocab_size, embedding_size], -1.0, 1.0),
                name="n_W")
            self.n_embedded_chars = tf.nn.embedding_lookup(self.n_W, self.input_n)
            self.n_embedded_chars_expanded = tf.expand_dims(self.n_embedded_chars, -1)

        with tf.name_scope("p_embedding"):
            self.p_W = tf.Variable(
                tf.random_uniform([p_vocab_size, embedding_size], -1.0, 1.0),
                name="p_W"
            )
            self.p_embedding_chars = tf.nn.embedding_lookup(self.p_W, self.input_p)  # shape (batch_size, code_length, statement_length, embedding_size)
            self.p_embedded_chars_expanded = tf.expand_dims(self.p_embedding_chars, -1) # shape (batch_size, code_length, statement_length, embedding_size, 1)

        # The first convolutional and pooling layer: represent the semantics of a statement based on the tokens within the statement
        with tf.name_scope("p_statement_conv_maxpool"):
            # Convolution Layer
            p_statement_filter_shape = [1, p_statement_filter_size, embedding_size, 1, p_statement_filters]
            p_statement_W = tf.Variable(tf.truncated_normal(p_statement_filter_shape, stddev=0.1), name="p_statement_W")
            p_statement_b = tf.Variable(tf.constant(0.1, shape=[p_statement_filters]), name="p_statement_b")
            p_statement_conv = tf.nn.conv3d(
                self.p_embedded_chars_expanded,
                p_statement_W,
                strides=[1, 1, 1, 1, 1],
                padding="VALID",
                name="p_statement_conv")  # shape (batch_size, code_length, statement_length-d+1, 1, p_statement_filters)
            # Apply nonlinearity
            p_statement_h = tf.nn.relu(tf.nn.bias_add(p_statement_conv, p_statement_b), name="p_statement_relu")
            # Maxpooling over the outputs
            p_statement_pooled = tf.nn.max_pool3d(
                p_statement_h,
                ksize=[1, 1, p_statement_length - p_statement_filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1, 1],
                padding='VALID',
                name="p_statement_pool") # shape (batch_size, code_length, 1, 1, p_statement_filters)
            self.p_squeezed_statement_pooled = tf.squeeze(p_statement_pooled, axis=[2, 3]) # shape (batch_size, code_length, p_statement_filters)
            self.p_statement_pooled_expanded = tf.expand_dims(self.p_squeezed_statement_pooled, -1) # shape (batch_size, code_length, p_statement_filters, 1)

        # The second convolutional and pooling layer: model the semantics conveyed by the interactions between statements
        p_semantics_pooled_outputs = []
        for i, p_semantics_filter_size in enumerate(p_semantics_filter_sizes):
            with tf.name_scope("p_semantics_conv_maxpool_%s" % p_semantics_filter_size):
                # Convolution Layer
                p_semantics_filter_shape = [p_semantics_filter_size, p_statement_filters, 1, p_semantics_filters]
                p_semantics_W = tf.Variable(tf.truncated_normal(p_semantics_filter_shape, stddev=0.1), name="p_semantics_W")
                p_semantics_b = tf.Variable(tf.constant(0.1, shape=[p_semantics_filters]), name="p_semantics_b")
                p_semantics_conv = tf.nn.conv2d(
                    self.p_statement_pooled_expanded,
                    p_semantics_W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="p_semantics_conv"
                )
                # Apply nonlinearity
                p_semantics_h = tf.nn.relu(tf.nn.bias_add(p_semantics_conv, p_semantics_b), name="p_semantics_relu")
                # Maxpooling over the outputs
                p_semantics_pooled = tf.nn.max_pool(
                    p_semantics_h,
                    ksize=[1, 100000, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="p_semantics_pooled")
                p_semantics_pooled_outputs.append(p_semantics_pooled)
                # Combine all the pooled features
                p_num_filters_total = p_semantics_filters * len(p_semantics_filter_sizes)
                self.p_h_pool = tf.concat(p_semantics_pooled_outputs, 3)
                self.p_h_pool_flat = tf.reshape(self.p_h_pool, [-1, p_num_filters_total])
