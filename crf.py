import tensorflow as tf


class CrfForwardRnnCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """Computes the alpha values in a linear-chain CRF.
    """

    def __init__(self, transition_params):
        """Initialize the CrfForwardRnnCell.
        """
        self._transition_params = transition_params
        self._num_tags = tf.shape(transition_params)[0]

        super(CrfForwardRnnCell, self).__init__()

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfForwardRnnCell.

        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous alpha
              values.
          scope: Unused variable scope of this cell.

        Returns:
          new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
              values containing the new alpha values.
        """
        # shape = (batch_size, num_tags, 1)
        state = tf.expand_dims(state, 2)

        # 按照博客的实现，感觉源码的实现好像有错误？？？
        # https://createmomo.github.io/2017/11/11/CRF-Layer-on-the-Top-of-BiLSTM-5/
        # shape = (batch_size, num_tags, num_tags)
        scores = state + self._transition_params + tf.expand_dims(inputs, [1])
        new_alphas = tf.reduce_logsumexp(scores, [1])

        # 源码实现
        # transition_scores = state + self._transition_params
        # new_alphas = inputs + tf.reduce_logsumexp(transition_scores, [1])

        return new_alphas, new_alphas


class LinearCRF:
    """线性链CRF"""

    def __init__(self, inputs, tag_indices, sequence_lengths, transition_params=None):
        """线性链条件随机场作为神经网络层

        Args:
          inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
              to use as input to the CRF layer.
          tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
              compute the unnormalized score.
          sequence_lengths: A [batch_size] vector of true sequence lengths.
          transition_params: A [num_tags, num_tags] transition matrix.
        Returns:
          sequence_scores: A [batch_size] vector of unnormalized sequence scores.
        """
        self.batch_size = tf.shape(inputs)[0]
        self.max_seq_len = tf.shape(inputs)[1]
        self.num_tags = tf.shape(inputs)[2]
        self.inputs = inputs
        self.tag_indices = tag_indices
        self.sequence_lengths = sequence_lengths

        # transition matrix
        if transition_params is None:
            num_tags = inputs.shape[2]
            self.transition_params = tf.get_variable("transitions", [num_tags, num_tags])
        else:
            self.transition_params = transition_params

    def log_likelihood(self):
        """计算对数似然
        """
        sequence_scores = self.sequence_score()
        log_norm = self.log_norm()
        log_likelihood = sequence_scores - log_norm
        return log_likelihood, self.transition_params

    def sequence_score(self):
        """对应源函数crf_sequence_score

        计算batch中各样本真实标记的分数, 分数定义为:
            scores = emission_scores + transition_scores
        源码的基本思想是: 通过一系列变换和设计辅助索引矩阵, 使得分数计算能通过矩阵实现,
            且不考虑padding的分数
        解释下各参数的对应关系:
            样本i的标记序列为: tags_i = tag_indices[i]
            样本i的第j个元素标记为: tags_i[j]
            样本i的第j个元素标记对应的emission score: inputs[i, j, tags_i[j]]
            样本i的第j个元素标记对应的transition score: transition_params[tags_i[j], tags_i[j + 1]]
        """
        # =======================crf_unary_score==================
        # 计算emission_scores
        # [batch_size * max_seq_len * num_tags]
        flattened_inputs = tf.reshape(self.inputs, [-1])
        offsets = tf.expand_dims(
            tf.range(self.batch_size) * self.max_seq_len * self.num_tags, 1)
        # [batch_size, 1] + [1, max_seq_len] -> [batch_size, max_seq_len]
        offsets += tf.expand_dims(tf.range(self.max_seq_len) * self.num_tags, 0)
        offsets = tf.cast(offsets, self.tag_indices.dtype)
        flattened_tag_indices = tf.reshape(offsets + self.tag_indices, [-1])
        # Gather slices from params axis axis according to indices
        unary_scores = tf.reshape(
            tf.gather(flattened_inputs, flattened_tag_indices),
            [self.batch_size, self.max_seq_len])
        masks = tf.sequence_mask(self.sequence_lengths,
                                 maxlen=tf.shape(self.tag_indices)[1],
                                 dtype=tf.float32)
        # [batch_size]
        unary_scores = tf.reduce_sum(unary_scores * masks, 1)

        # ======================crf_binary_score=================
        # 计算transition_scores
        # 标签可转移数量等于最大时间步减1
        num_transitions = tf.shape(self.tag_indices)[1] - 1
        # [batch_size, num_transitions]
        start_indices = tf.slice(self.tag_indices, [0, 0], [-1, num_transitions])
        end_indices = tf.slice(self.tag_indices, [0, 1], [-1, num_transitions])
        flattened_trans_indices = start_indices * self.num_tags + end_indices
        # [num_tags * num_tags]
        flattened_trans_params = tf.reshape(self.transition_params, [-1])
        # [batch_size, num_transitions]
        binary_scores = tf.gather(flattened_trans_params, flattened_trans_indices)
        truncated_masks = tf.slice(masks, [0, 1], [-1, -1])
        # [batch_size]
        binary_scores = tf.reduce_sum(binary_scores * truncated_masks, 1)

        sequence_scores = unary_scores + binary_scores
        return sequence_scores

    def log_norm(self):
        """对应源函数crf_log_norm
        Computes the normalization for a CRF.
        """
        # [batch_size, 1, num_tags]
        first_input = tf.slice(self.inputs, [0, 0, 0], [-1, 1, -1])
        # [batch_size, num_tags], 每个batch的第一个词的发射向量
        first_input = tf.squeeze(first_input, [1])
        # [batch_size, max_seq_len - 1, num_tags]
        rest_of_input = tf.slice(self.inputs, [0, 1, 0], [-1, -1, -1])

        forward_cell = CrfForwardRnnCell(self.transition_params)
        sequence_lengths = tf.maximum(
            tf.constant(0, dtype=self.sequence_lengths.dtype),
            self.sequence_lengths - 1)
        outputs, state = tf.nn.dynamic_rnn(
            cell=forward_cell,
            inputs=rest_of_input,
            sequence_length=sequence_lengths,
            initial_state=first_input,
            dtype=tf.float32)
        log_norm = tf.reduce_logsumexp(state, [1])
        log_norm = tf.where(
            tf.less_equal(sequence_lengths, 0), tf.zeros_like(log_norm), log_norm)
        return log_norm
