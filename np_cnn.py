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

        # Convolution for natural text
        # Embedding layer
        with tf.name_scope("n_embedding"):
            self.n_W = tf.Variable(
                tf.random_uniform([n_vocab_size, embedding_size], -1.0, 1.0),
                name="n_W")
            self.n_embedded_chars = tf.nn.embedding_lookup(self.n_W, self.input_n) # shape (batch_size, n_sentence_length, embedding_size)
            self.n_embedded_chars_expanded = tf.expand_dims(self.n_embedded_chars, -1) # shape (batch_size, n_sentence_length, embedding_size, 1)

        # Create a convolution + maxpool layer for each filter size
        n_pooled_outputs = []
        for i, filter_size in enumerate(n_filter_sizes):
            with tf.name_scope("n_conv_maxpool_%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, n_num_filters]
                n_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="n_W")
                n_b = tf.Variable(tf.constant(0.1, shape=[n_num_filters]), name="n_b")
                n_conv = tf.nn.conv2d(
                    self.n_embedded_chars_expanded,
                    n_W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="n_conv") # shape (batch_size, n_sentence_length - fiter_size + 1, 1 fiters)
                # Apply nonlinearity
                n_h = tf.nn.relu(tf.nn.bias_add(n_conv, n_b), name="n_relu") # shape (batch_size, n_sentence_length - fiter_size + 1, 1 fiters)
                # Maxpooling over the outputs
                n_pooled = tf.nn.max_pool(
                    n_h,
                    ksize=[1, n_sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="n_pool") # shape (batch_size, 1, 1, fiters)
                n_pooled_outputs.append(n_pooled)
        # Combine all the natural pooled features
        n_num_filters_total = n_num_filters * len(n_filter_sizes)
        self.n_h_pool = tf.concat(n_pooled_outputs, 3) # shape (batch_size, 1, 1, fiters * len(n_filter_sizes))
        self.n_h_pool_flat = tf.reshape(self.n_h_pool, [-1, n_num_filters_total]) # shape (batch_size, filters * num_fiters)

        # Convolution for program
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
                    name="p_semantics_conv") # shape (batch_size, code_length, 1, p_semantics_filters)
                # Apply nonlinearity
                p_semantics_h = tf.nn.relu(tf.nn.bias_add(p_semantics_conv, p_semantics_b), name="p_semantics_relu")
                # Maxpooling over the outputs
                p_semantics_pooled = tf.nn.max_pool(
                    p_semantics_h,
                    ksize=[1, 100000, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="p_semantics_pooled") # shape (batch_size, 1, 1, p_semantics_filters)
                p_semantics_pooled_outputs.append(p_semantics_pooled)
        # Combine all the pooled features
        p_num_filters_total = p_semantics_filters * len(p_semantics_filter_sizes)
        self.p_h_pool = tf.concat(p_semantics_pooled_outputs, 3) # shape (batch_size, 1, 1, p_semantics_filters * len(p_filter_sizes))
        self.p_h_pool_flat = tf.reshape(self.p_h_pool, [-1, p_num_filters_total]) # shape (batch_size, p_semantics_filters * len(p_filter_sizes))

        # Cross-language Feature Fusion Layers
        with tf.name_scope("fusion"):
            np_fusion_num_filters = n_num_filters_total + p_num_filters_total
            self.np_h_fusion = tf.concat([self.n_h_pool_flat, self.p_h_pool_flat], 1) # shape (batch_size, n_h_pool_flat.shape[1] + p_h_pool_flat.shape[1])

        # Add dropout
        with tf.name_scope("dropout"):
            self.np_h_drop = tf.nn.dropout(self.np_h_fusion, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[np_fusion_num_filters, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.np_h_drop, W, b, name="logits")
            self.scores = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

            # Calculate mean cross-entropy loss
            with tf.name_scope("loss"):
                if add_class_weights:
                    # classes_weights = tf.constant([0.57, 0.43])
                    # losses = tf.nn.weighted_cross_entropy_with_logits(logits=self.scores, targets=self.input_y,
                    #                                                   pos_weight=classes_weights)
                    if class_weights is None:
                        weights = tf.reduce_sum(self.input_y, axis=1)
                    else:
                        weights = tf.reduce_sum(class_weights * self.input_y, axis=1)
                    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                                logits=self.logits)
                    weighted_losses = unweighted_losses * weights
                    losses = tf.reduce_mean(weighted_losses)
                else:
                    losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            with tf.name_scope("evaluate_measures"):
                # Accuracy
                with tf.name_scope("accuracy"):
                    self.labels = tf.argmax(self.input_y, 1)
                    correct_predictions = tf.equal(self.predictions, self.labels)
                    self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

                p_cond = tf.equal(self.labels, 0)  # get positive answer position, return a list of True and False, the p position corresponds to True
                n_cond = tf.equal(self.labels, 1)  # get negative answer position
                p_prediction = tf.boolean_mask(self.predictions, p_cond)
                p_label = tf.boolean_mask(self.labels, p_cond)
                self.tp = tf.reduce_sum(tf.cast(tf.equal(p_prediction, p_label), "float"), name="tp")
                self.fp = tf.subtract(tf.reduce_sum(tf.cast(tf.equal(self.predictions, 0), "float")), self.tp)
                n_prediction = tf.boolean_mask(self.predictions, n_cond)
                n_label = tf.boolean_mask(self.labels, n_cond)
                self.tn = tf.reduce_sum(tf.cast(tf.equal(n_prediction, n_label), "int64"), name="tn")
                self.fn = tf.subtract(tf.reduce_sum(self.predictions), self.tn)
                # cast tp, fp, tn, fn to float64
                self.tp = tf.cast(self.tp, tf.float64)
                self.fp = tf.cast(self.fp, tf.float64)
                self.tn = tf.cast(self.tn, tf.float64)
                self.fn = tf.cast(self.fn, tf.float64)
                # Precision
                self.precision = tf.div(self.tp, tf.add(self.tp, self.fp), name="precision")
                # Recall
                self.recall = tf.div(self.tp, tf.add(self.tp, self.fn), name="recall")
                # F1-score
                self.product = tf.multiply(tf.cast(tf.constant(2.0), tf.float64),
                                           tf.multiply(self.precision, self.recall))
                self.f1_score = tf.div(self.product, tf.add(self.precision, self.recall), name="f1_score")



