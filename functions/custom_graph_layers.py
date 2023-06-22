"""
Custom layers and functions for simple graph operations.
"""
from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer, Softmax

from .custom_base_layers import FCDenseBlock


class CreateMask(Layer):
    """Create a boolean mask for existing jets"""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            inputs:
                Inputs to the layer for which a mask should be created, typically the
                jets.
                Expected shape: (batch_size, n_nodes, n_input_features)

        Returns
        -------
            mask:
                Mask being one for existing jets and zero for non-existent jets.
                Expected shape: (batch_size, n_nodes, 1)

        """
        non_zero_inputs = tf.where(tf.abs(inputs[:, :, :4]) > 1e-8, 1.0, 0.0)
        mask = tf.cast(
            tf.reduce_sum(non_zero_inputs, axis=2) == 4,
            tf.float32,
        )

        return mask


def expand_nodes_with_mask(inputs: tf.Tensor, mask: tf.Tensor):
    """
    Return a tensor with 3 dimensions which was reduced by tf.boolean_mask() back to its
    original shape

    Args
    ----
        inputs:
            Masked tensor
        mask:
            Mask

    Returns
    -------
        outputs:
            3 dimensional tensor after the unmasking. The first two dimensions will
            be the same as of the mask tensor. The previously masked entries are
            set to zeros.

    """
    if mask is None:
        return inputs
    scatter_shape = [
        tf.keras.backend.shape(mask)[0],
        mask.shape[1],
        tf.keras.backend.int_shape(inputs)[-1],
    ]

    outputs = tf.scatter_nd(tf.where(mask), inputs, scatter_shape)
    return outputs


def expand_edges_with_mask(inputs: tf.Tensor, mask: tf.Tensor):
    """
    Return a tensor with 4 dimensions which was reduced by tf.boolean_mask() back to its
    original shape

    Args
    ----
        inputs:
            Masked tensor
        mask:
            Mask

    Returns
    -------
        outputs:
            4 dimensional tensor after the unmasking. The first three dimensions will
            be the same as of the mask tensor. The previously masked entries are
            set to zeros.

    """
    if mask is None:
        return inputs

    scatter_shape = [
        tf.keras.backend.shape(mask)[0],
        mask.shape[1],
        mask.shape[2],
        tf.keras.backend.int_shape(inputs)[-1],
    ]

    outputs = tf.scatter_nd(tf.where(mask), inputs, scatter_shape)
    return outputs


def pass_with_mask(
    inputs: tf.Tensor, lyr: Layer, mask: Optional[tf.Tensor]
) -> tf.Tensor:
    """
    Pass some inputs through a tf layer. If not all objects of the input are present,
    the non-existing objects are removed before passing them through the layer to save
    computation time. Afterwards, the inputs are expanded back to their original shape
    (apart from the last dimension which can be changed by the layer).

    Args
    ----
        inputs:
            Inputs which should be passed through a tf layer. Needs to be (at least)??
            of rank 3 (batch, objects, features), otherwise masking doesn't make sense
        lyr:
            Layer object that the inputs will be passed through. E.g. a simple dense
            layer.
        mask:
            Mask for non-existing objects. If no mask is given, the inputs are just
            passed through the layer.

    Returns
    -------
        full_outputs:
            Inputs after passing them through the given layer. If a mask is given,
            non-existing objects will have 0 for all their features.

    """
    if mask is None:
        return lyr(inputs)
    masked_outputs = lyr(tf.boolean_mask(inputs, tf.cast(mask, tf.bool)))
    full_outputs = expand_nodes_with_mask(masked_outputs, mask)
    return full_outputs


class GraphBlock(Layer):
    """
    Create a graph block processing a single input.
    """

    def __init__(
        self,
        dense_config_edges: dict,
        dense_config_nodes: dict,
        k_neighbours: int = 15,
        pooling_edges: str = "avg",
        attention_network: Optional[dict] = None,
        **kwargs,
    ):
        """
        Create a graph block processing a single input. The block consists of building
        edges between the input objects, passing the edges through a dense network,
        pooling the processed edges, concatenate the pooled edges with the original
        inputs, and pass the concatenated thing through a dense network.

        Args
        ----
            dense_config_edges:
                Configuration dictionary for the dense network processing the edges.
                'FCDenseBlock' is used to create the dense network.
            dense_config_nodes:
                Configuration dictionary for the dense network processing the nodes.
                'FCDenseBlock' is used to create the dense network.
            k_neighbours:
                Number of k nearest neighbours used while building edges
            pooling_edges:
                String to define which pooling operation is used to pool the edges.
            attention_network:
                Configuration dictionary defining a network predicting attention weights
                for the single edges for attention pooling.

        """
        super().__init__(**kwargs)
        self.edge_block = EdgeBlock(
            dense_edges=dense_config_edges,
            pooling_edges=pooling_edges,
            attention_network=attention_network,
            k_neighbours=k_neighbours,
        )

        self.dense_block_nodes = FCDenseBlock(dense_config_nodes)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor = None) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            inputs:
                Original nodes being processed by this graph block
                Expected shape: (batch_size, n_nodes, n_input_features)
            mask:
                Mask being one for existing nodes and zero for non-existing nodes.
                Expected shape: (batch_size, n_nodes, 1)


        Returns
        -------
            nodes:
                Nodes after being processed by the GraphBlock
                Expected shape: (batch_size, n_nodes, n_output_features)

        """
        pooled_edges, _ = self.edge_block(inputs, inputs, mask, mask, None)
        nodes_plus_pooled_edges = tf.concat([inputs, pooled_edges], axis=-1)
        # Pass inputs + pooled edges through dense net, again with masks applied
        nodes = pass_with_mask(nodes_plus_pooled_edges, self.dense_block_nodes, mask)

        return nodes


def knn(input_x, input_y, k_neighbours, self_connection=False):
    """
    Calculate k nearest neighbors and the distance between the objects using L2 norm.

    Args
    ----
        input_x:
            First set of objects
        input_y:
            Second set of objects
        k_neighbours:
            Number of neighbours
        self_connection:
            Whether to include the closest neighbour or not If the same input is given
            twice, the closest neighbour is always the same object leading to a
            connection to itself if the closest neighbour is included.

    Returns
    -------
        distances:
            Distance between all objects
        indx:
            Indices of the nearest neighbours.

    """
    distances = euclidean_squared(input_x, input_y)
    if self_connection:
        _, indx = tf.math.top_k(-distances, k_neighbours + 1)
        return distances, indx[..., 1:]

    _, indx = tf.math.top_k(-distances, k_neighbours)
    return distances, indx


class PoolingLayerMax(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        return tf.reduce_max(inputs, axis=1)


class PoolingLayerSum(Layer):
    def call(self, inputs, mask=None):
        return tf.reduce_sum(inputs, axis=1)


class PoolingLayerAvg(Layer):
    def call(self, inputs, mask=None):
        if mask is None:
            return tf.math.reduce_mean(inputs, axis=1)

        return tf.math.reduce_sum(inputs, axis=1) / (
            tf.math.reduce_sum(mask[..., None, None], axis=1) + 1e-10
        )


class PoolingLayerAtt(Layer):
    def call(self, inputs, mask=None):
        weighted = inputs[0] * inputs[1]
        return tf.math.reduce_sum(weighted, axis=1)


class PoolingLayer(Layer):
    def __init__(self, pooling: str, **kwargs):
        super().__init__(**kwargs)
        self.pooling = pooling
        if self.pooling == "max":
            self.pool_layer = PoolingLayerMax()
        elif self.pooling == "sum":
            self.pool_layer = PoolingLayerSum()
        elif self.pooling == "avg":
            self.pool_layer = PoolingLayerAvg()
        elif self.pooling == "att":
            self.pool_layer = PoolingLayerAtt()
        else:
            raise NotImplementedError(
                f'Pooling of type "{self.pooling}" not implemented.'
            )

    def call(self, inputs, mask=None):
        return self.pool_layer(inputs, mask)


def euclidean_squared(input_x, input_y):
    """
    Taken from https://github.com/jkiesele/caloGraphNN/blob/master/caloGraphNN.py

    Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F]
    where B is batch size, N is number of examples in the batch of first set,
    M is number of examples in the batch of second set, F is number of spatial features.

    Args
    ----
        input_x:
            First set of objects
        input_y:
            Second set of objects

    Returns
    -------
        A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance
        between ith entry in first set and jth in second set.

    """

    shape_x = input_x.get_shape().as_list()
    shape_y = input_y.get_shape().as_list()

    assert (input_x.dtype in (tf.float16, tf.float32, tf.float64)) and (
        input_y.dtype in (tf.float16, tf.float32, tf.float64)
    )
    assert len(shape_x) == 3 and len(shape_y) == 3
    assert shape_x[0] == shape_y[0]

    # Finds euclidean distance using property (x-y)^2 = x^2 + y^2 - 2xy
    mixed_term = -2 * tf.matmul(
        input_x, tf.transpose(input_y, perm=[0, 2, 1])
    )  # -2xb term
    x_squared = tf.expand_dims(
        tf.reduce_sum(input_x * input_x, axis=2), axis=2
    )  # x^2 term
    y_squared = tf.expand_dims(
        tf.reduce_sum(input_y * input_y, axis=2), axis=1
    )  # y^2 term
    return tf.abs(mixed_term + x_squared + y_squared)


class AttentionLayer(Layer):
    """
    Calculate attention weights for attention pooling.
    """

    def __init__(self, attention_architecture: dict, masked: bool = False, **kwargs):
        """
        Calculate attention weights for attention pooling.

        Args
        ----
            attention_architecture:
                Configuration of the dense network used to obtain the attention weights.
                An additional linear layer with a single output neuron and a linear
                activation function is always added at the end.
            masked:
                Whether the input passed to the layer is masked, i.e. has non-existing
                nodes or edges are removed or not.

        Returns
        -------
            Attention weights to be used for the pooling.

        """
        super().__init__(**kwargs)
        self.attention_architecture = attention_architecture
        self.masked = masked
        self.dense = FCDenseBlock(
            attention_architecture, {"units": 1, "activation": "linear"}, name=""
        )

    def call(self, inputs, mask=None):
        """
        Call the layer

        Args
        ----
            inputs:
                Nodes or edges to perform attention pooling over. Attention weights will
                be calculated for these objects
            mask:
                Mask for non-existing edges/nodes

        Returns
        -------
            Attention weights to be used for the pooling.

        """
        if self.masked:
            tns = self.dense(inputs)
            tns = expand_edges_with_mask(tns, mask)
        else:
            tns = pass_with_mask(inputs, self.dense, mask)
        tns = tns - 999.0 * (1 - mask[..., None])

        return Softmax(axis=1)(tns)


class EdgeBlock(Layer):
    """
    Form edges between nodes, pass them through a dense network, and pool them.
    """

    def __init__(
        self,
        dense_edges: dict,
        pooling_edges: str = "avg",
        attention_network: Optional[dict] = None,
        k_neighbours: int = 1,
        aggregation: str = "concat",
        single_connection: bool = False,
        fully_connected: bool = False,
        **kwargs,
    ):
        """
        Form edges between nodes, pass them through a dense network, and pool them.

        Args
        ----
            dense_edges:
                Configuration of the dense network processing the edges.
            pooling_edges:
                Pooling operation used to pool the edges.
            attention_network:
                If attention pooling is wanted, configuration for the attention network.
            k_neighbours:
                Number of k-nearest neighbours to use when building edges (if
                'single_connection' of 'fully_connected' aren't set).
            aggregation:
                The way the edges are aggregated from the nodes.
            single_connection:
                Whether to use an identity matrix as sender matrix.
            fully_connected:
                Whether to use a matrix with all entries equal to one as sender matrix.

        """
        super().__init__(**kwargs)
        self.aggregation_ops = EdgeAggregation(
            k_neighbours, aggregation, single_connection, fully_connected
        )
        self.fully_connected = fully_connected
        self.single_connection = single_connection
        self.dense_net = FCDenseBlock(dense_edges)
        self.pooling_layer = PoolingLayer(pooling=pooling_edges)
        self.do_attention = False
        if pooling_edges == "att":
            self.do_attention = True
            self.attention_net = AttentionLayer(attention_network, masked=True)

    def call(
        self,
        receivers,
        senders,
        mask_receivers=None,
        mask_senders=None,
        previous_edges=None,
    ):
        """
        Call the layer

        Args
        ----
            receivers:
                Nodes which receive a message
            senders:
                Nodes which send a message
            mask_receivers:
                Mask for non-existing receivers
            mask_senders:
                Mask for non-existing senders
            previous_edges:
                Previous edge features to be concatenated to the aggregated edges before
                being passed through the dense network.

        Returns
        -------
            pooled_edges:
                Pooled edges
            updated_edges:
                New persistent edge features

        """
        edges, mask_edges = self.aggregation_ops(
            receivers, senders, mask_receivers, mask_senders
        )
        if previous_edges is not None:
            edges = tf.concat([edges, previous_edges], axis=-1)
        updated_edges = self.dense_net(edges)
        full_edges = expand_edges_with_mask(updated_edges, mask_edges)
        if self.do_attention:
            att_edges = self.attention_net(edges, mask=mask_edges)
            # Pool over the edges
            pooled_edges = self.pooling_layer([att_edges, full_edges], mask_receivers)
        else:
            pooled_edges = self.pooling_layer(full_edges, mask_receivers)

        return pooled_edges, updated_edges


class EdgeAggregation(Layer):
    """
    Construct edges between two types of nodes (can be the same nodes twice) based on
    k-nearest neighbours.
    """

    def __init__(
        self,
        k_neighbours: int = 1,
        aggregation: str = "concat",
        single_connection: bool = False,
        fully_connected: bool = False,
        **kwargs,
    ):
        """
        Construct edges between two types of nodes (can be the same nodes twice) based
        on k-nearest neighbours.

        Args
        ----
            k_neighbours:
                Number of neighbours for each node unless one of 'single_connection' or
                'fully_connected' is set to True.
            aggregation:
                How to aggregate the edges based on the node features. 'concat'
                concatenates the node features, 'subtract' subtracts them from each
                other, and 'both' concatenates them and subtracts them from each other.
                For 'subtract' and 'both', senders and receivers have to have the same
                number of features (last dimension).
            single_connection:
                If set to True, the sender matrix will be constructed without a kNN but
                as an identity matrix. Can't be set to True at the same time as
                'fully_connected'.
            full_connected:
                If set to True, the sender matrix will be constructed without a kNN but
                as a matrix with all ones. So each node sends to each other node
                (self-interaction included if sender and receivers are the same nodes).
                 Can't be set to True at the same time as 'single_connection'.

        """
        super().__init__(**kwargs)
        self.k_neighbours = k_neighbours
        self.aggregation = aggregation
        self.fully_connected = fully_connected
        self.single_connection = single_connection

        if self.fully_connected and self.single_connection:
            raise ValueError(
                "Both 'fully_connected' and 'single_connection' set to True."
                + " Don't know what to do..."
            )

    def call(self, receivers, senders, mask_receivers=None, mask_senders=None):
        """
        Call the layer

        Args
        ----
            receivers:
                Nodes that receive the message
                Expected shape: (batch_size, n_receivers, n_features_r)
            sender:
                Nodes that send the message. Can be the same as 'receivers'.
                Expected shape: (batch_size, n_senders, n_features_s)
            mask_receivers:
                Mask for existing and non-existing receiver nodes.
                Expected shape: (batch_size, n_receivers, 1)
            mask_senders:
                Mask for existing and non-existing sender nodes.
                Expected shape: (batch_size, n_senders, 1)

        Returns
        -------
            edges:
                Formed edges between senders and receivers
                Expected shape: (batch_size, n_senders, n_receivers,
                                 n_features_r+n_features_s ('concat'),
                                 n_features_r ('subtract'),
                                 3 * n_features_r ('both'))
            sender:
                Sender matrix between senders and receivers
                Expected shape: (batch_size, n_senders, n_receivers)

        """
        n_batch = tf.keras.backend.shape(senders)[0]
        n_senders = tf.keras.backend.shape(senders)[1]
        n_receivers = tf.keras.backend.shape(receivers)[1]

        if mask_senders is None:
            knn_senders = senders
        else:
            knn_senders = senders - 99 * (1 - tf.expand_dims(mask_senders, -1))
        if mask_receivers is None:
            knn_receivers = receivers
        else:
            knn_receivers = receivers - 99 * (1 - tf.expand_dims(mask_receivers, -1))

        if self.fully_connected:
            sender = tf.ones(shape=(n_batch, n_senders, n_receivers))
        elif self.single_connection:
            sender = tf.eye(
                num_rows=n_senders,
                num_columns=n_receivers,
                batch_shape=[n_batch],
                dtype="float32",
            )
        else:
            _, indx = knn(knn_senders, knn_receivers, self.k_neighbours)
            sender = tf.reduce_sum(tf.one_hot(indx, n_receivers), axis=-2)

        if mask_senders is not None:
            sender = sender * mask_senders[..., None]
        if mask_receivers is not None:
            sender = sender * mask_receivers[..., None, :]

        sending = tf.expand_dims(sender, -1) * tf.expand_dims(senders, -2)
        sending = tf.boolean_mask(sending, sender)
        receiving = tf.expand_dims(sender, -1) * tf.expand_dims(receivers, -3)
        receiving = tf.boolean_mask(receiving, sender)

        if self.aggregation == "concat":
            edges = tf.concat([sending, receiving], axis=-1)
        elif self.aggregation == "subtract":
            edges = sending - receiving

        elif self.aggregation == "both":
            edges = tf.concat(
                [sending, receiving, sending - receiving],
                axis=-1,
            )

        return edges, sender
