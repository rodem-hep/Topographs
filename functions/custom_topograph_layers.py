"""
Custom layers and functions for the topograph model.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer

from .custom_base_layers import FCDenseBlock
from .custom_graph_layers import (
    AttentionLayer,
    EdgeAggregation,
    EdgeBlock,
    PoolingLayer,
    expand_edges_with_mask,
    pass_with_mask,
)


@dataclass
class TopoGraphData:
    """
    Class to hold all data needed for the topograph model, containing jets, a mask for
    existing/non-existing jets, helper nodes for ws and tops, and previous edges.
    """

    jets: tf.Tensor
    mask: tf.Tensor
    nodes_w: tf.Tensor = None
    nodes_top: tf.Tensor = None
    previous_edges: List[tf.Tensor] = None


class ParticlePredictionMLPs(Layer):
    """
    Create MLPs for regression for multiple particles of the same type at the same time.
    """

    def __init__(self, architecture: dict, n_particles: int = 2, **kwargs):
        """
        Create MLPs for regression for multiple particles of the same type at the same
        time. For each particle a separate MLP with the same HPs will be created and
        used for regression.

        Args
        ----
            architecture:
                Dictionary holding the configuration of the MLPs. To be used with
                'FCDenseBlock'. Additionally, an 'out' key is needed to specify the
                number of outputs of the regression networks. No activation is used for
                the final linear layer.
            n_particles:
                Number of particles. That many MLPs will be created.

        """
        super().__init__(**kwargs)
        self.n_particles = n_particles
        self.dense_blocks = [
            FCDenseBlock(
                architecture, {"units": architecture["out"], "activation": "linear"}
            )
            for _ in range(self.n_particles)
        ]

    def call(self, tns: List[tf.Tensor]) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            tns:
                List of tensors where each tensor corresponds to one particle/node, i.e.
                the length of the list should be 'self.n_particles'.
                Expected shape of each tensor: (batch_size, n_input_features).
                Additional dimensions can be included.

        Returns
        -------
            Regression results for all particles concatenated into one tensor.
            Expected shape: (batch_size, n_particles, architecture['out']). Additional
            dimensions can be included.

        """
        outputs = []
        for i, dense_block in enumerate(self.dense_blocks):
            dense = dense_block(tns[:, i])
            outputs.append(tf.expand_dims(dense, 1))

        return tf.concat(outputs, axis=1)


class InitializeHelperNodesW(Layer):
    """
    Initialize the helper nodes representing the W bosons from the jets.
    """

    def __init__(
        self,
        first_w_initialization: str,
        second_w_initialization: str,
        first_attention_architecture: Optional[dict] = None,
        second_attention_architecture: Optional[dict] = None,
        **kwargs,
    ):
        """
        Initialize the helper nodes representing the W bosons from the jets.

        Args
        ----
            first_w_initialization:
                How the first helper node is initialized. All pooling opertaions are
                allowed.
            second_w_initialization:
                How the second helper node is initialized. On top of all pooling
                operations, 'opp' can be used which initializes the second node as the
                first one multiplied with negative one.
            first_attention_architecture:
                If attention pooling is wanted for the first helper node, the
                architecture of the attention network can be given here.
            second_attention_architecture:
                If attention pooling is wanted for the second helper node, the
                architecture of the attention network can be given here.

        """
        super().__init__(**kwargs)
        self.first_w_initialization = first_w_initialization
        self.second_w_initialization = second_w_initialization

        if self.first_w_initialization == "att":
            self.first_attention_layer = AttentionLayer(first_attention_architecture)
        self.first_pooling_layer = PoolingLayer(self.first_w_initialization)

        if self.second_w_initialization == "att":
            self.second_attention_layer = AttentionLayer(second_attention_architecture)
        if self.second_w_initialization != "opp":
            self.second_pooling_layer = PoolingLayer(self.second_w_initialization)

    def call(self, inputs: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        """
        Call the layer.

        Args
        ----
            inputs:
                Nodes/Jets to be pooled over to initialize the helper nodes.
                Expected shape: (batch_size, n_nodes, n_features)
            mask:
                Mask for existing/non-existing nodes/jets.
                Expected shape: (batch_size, n_nodes, 1)

        Returns
        -------
            Initialized helper nodes.
            Expected shape: (batch_size, 2, n_features)

        """
        if hasattr(self, "first_attention_layer"):
            attention_weights = self.first_attention_layer(inputs, mask)
            first_pooled_nodes = self.first_pooling_layer(
                [inputs, attention_weights], mask
            )
        else:
            first_pooled_nodes = self.first_pooling_layer(inputs, mask)

        if hasattr(self, "second_attention_layer"):
            attention_weights = self.second_attention_layer(inputs, mask)
            second_pooled_nodes = self.second_pooling_layer(
                [inputs, attention_weights], mask
            )
        elif hasattr(self, "second_pooling_layer"):
            second_pooled_nodes = self.second_pooling_layer(inputs, mask)
        else:
            second_pooled_nodes = -first_pooled_nodes

        return tf.concat(
            [first_pooled_nodes[:, None, :], second_pooled_nodes[:, None, :]], axis=1
        )


class InitializeHelperNodesTopFixed(Layer):
    """
    Initialize the helper nodes representing the top bosons from the jets and the W
    nodes.
    """

    def __init__(
        self,
        jets_pooling: str,
        attention_net_architecture: Optional[dict],
        **kwargs,
    ):
        """
        Initialize the helper nodes representing the top bosons from the jets and the W
        nodes. The jets are pooled in some configurable way and the W node is
        concatenated onto it. This fixes a connection between the W nodes and the top
        nodes.

        Args
        ----
            jets_pooling:
                Pooling to be applied to the jets to get part of the to be initialized
                helper nodes.
            attention_net_architecture:
                If attention pooling is wanted, architecture of the attention network

        """
        super().__init__(**kwargs)

        if jets_pooling == "att":
            self.first_attention_net = AttentionLayer(attention_net_architecture)
            self.second_attention_net = AttentionLayer(attention_net_architecture)
        # Pooling layers don't have parameters, so the same can be used for both tops
        self.jets_pooling_layer = PoolingLayer(jets_pooling)

    def call(self, inputs: List[tf.Tensor], mask: tf.Tensor) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            inputs:
                List of the inputs to initialize the helper nodes. The first entry is
                for the jet nodes and the second for the W nodes.
                Expected shape: [(batch_size, n_nodes, n_features_1),
                                 (batch_size, 2, n_features_2)]
            mask:
                Mask for existing/non-existing nodes/jets.
                Expected shape: (batch_size, n_nodes, 1)

        Returns
        -------
            Initialized helper nodes.
            Expected shape: (batch_size, 2, n_features_1 + n_features_2)

        """
        if hasattr(self, "first_attention_net"):
            first_attention_weights = self.first_attention_net(inputs[0], mask)
            second_attention_weights = self.second_attention_net(inputs[0], mask)

            first_pooled_jets = self.jets_pooling_layer(
                [inputs[0], first_attention_weights], mask
            )
            second_pooled_jets = self.jets_pooling_layer(
                [inputs[0], second_attention_weights], mask
            )
        else:
            first_pooled_jets = self.jets_pooling_layer(inputs[0], mask)
            second_pooled_jets = self.jets_pooling_layer(inputs[0], mask)

        first_top = tf.concat([first_pooled_jets, inputs[1][:, 0]], axis=-1)
        second_top = tf.concat([second_pooled_jets, inputs[1][:, 1]], axis=-1)

        return tf.concat([first_top[:, None, :], second_top[:, None, :]], axis=1)


class EdgeClassification(Layer):
    """
    Form edges between nodes and classify them.
    """

    def __init__(self, classification_net: dict, **kwargs):
        """
        Form edges between nodes and classify them.

        Args
        ----
            classification_net:
                Architecture of the network to classify the edges. A final layer with
                one output and a sigmoid activation is added at the end.

        """
        super().__init__(**kwargs)
        self.classification_net = classification_net
        self.edge_aggregation = EdgeAggregation(fully_connected=True)
        self.classification_net = FCDenseBlock(
            classification_net, {"units": 1, "activation": "sigmoid"}
        )

    def call(self, inputs: List[tf.Tensor], mask: tf.Tensor) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            inputs:
                Nodes between which the edges should be classified. The first entry
                in the list should be the jet nodes and the second the W/top nodes.
                Expected shape: [(batch_size, n_nodes_type_1, n_features_1),
                                 (batch_size, n_nodes_type_2, n_features_2)]
                                 (n_features_i doesn't have any impact on this
                                 implementation)

            mask:
                Mask of existing and non-existing nodes for the first entry in the input
                list. It is assumed that all nodes for the second entry in the input
                list exist.
                Expected shape: (batch_size, n_nodes_type_1, 1)

        Returns
        -------
            edge_scores:
                Score for every edge between the two types of nodes.
                Expected shape: (batch_size, n_nodes_type_1, n_nodes_type_2, 1)

        """
        edges_jet_particle, mask_jet_particle = self.edge_aggregation(
            inputs[1], inputs[0], None, mask
        )

        edge_scores = self.classification_net(edges_jet_particle)
        edge_scores = expand_edges_with_mask(edge_scores, mask_jet_particle)

        return edge_scores


class TopoGraphBlock(Layer):
    """
    Class defining a Topograph block.
    """

    def __init__(
        self,
        topo_graph_layer_kwargs: dict,
        classification_net: dict,
        regression_net: dict,
        **kwargs,
    ):
        """
        Class defining a Topograph block: a TopographLayer for information exchange
        between all nodes, two EdgeClassification layers to classify edges to the W and
        top nodes, and two ParticlePredictionMLPs to regress towards the true parton
        properties for the W and top nodes.

        Args
        ----
            topo_graph_layer_kwargs:
                Configuration for the TopoGraphLayer in dictionary form.
            classification_net:
                Configuration for the edge classification networks. All edge
                classification networks in one block have the same HPs.
            regression_net:
                Configuration for the regression networks. All regression networks in
                one block have the same HPs.

        """
        super().__init__(**kwargs)
        self.topo_graph_layer = TopoGraphLayer(**topo_graph_layer_kwargs)
        self.classification_w_layer = EdgeClassification(
            classification_net=classification_net
        )
        self.classification_top_layer = EdgeClassification(
            classification_net=classification_net
        )
        self.regression_w_layer = ParticlePredictionMLPs(
            architecture=regression_net, n_particles=2
        )
        self.regression_top_layer = ParticlePredictionMLPs(
            architecture=regression_net, n_particles=2
        )

    def call(
        self, data: TopoGraphData
    ) -> Tuple[TopoGraphData, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Call the layer

        Args
        ----
            data:
                TopoGraphData holding the jet nodes, W nodes, top nodes, mask for
                existing/non-existing jets, and potentially previous edge features.

        Returns
        -------
            data:
                Updated TopographData. Will update everything apart from the mask.
            regression_w:
                Regression results for the two W nodes.
                Expected shape: (batch_size, 2, n_regression_features)
            regression_top:
                Regression results for the two top nodes.
                Expected shape: (batch_size, 2, n_regression_features)
            classification_w:
                Edge classification results for the two W nodes.
                Expected shape: (batch_size, n_jets, 2, 1)
            classification_top:
                Edge classification results for the two top nodes.
                Expected shape: (batch_size, n_jets, 2, 1)

        """
        data = self.topo_graph_layer(data)
        regression_w = self.regression_w_layer(data.nodes_w)
        regression_top = self.regression_top_layer(data.nodes_top)
        classification_w = self.classification_w_layer(
            [data.jets, data.nodes_w], data.mask
        )

        classification_top = self.classification_top_layer(
            [data.jets, data.nodes_top], data.mask
        )

        return data, regression_w, regression_top, classification_w, classification_top


class TopoGraphLayer(Layer):
    """
    A single TopoGraphLayer which passes messages between all different types of nodes.
    """

    def __init__(
        self,
        edge_net_architecture: dict,
        node_net_architecture: dict,
        pooling: str = "avg",
        attention_net_architecture: Optional[dict] = None,
        full_connections_jets: bool = False,
        full_connections_ws: bool = False,
        full_connections_tops: bool = False,
        full_connections_w_top: bool = False,
        w_w_interaction: bool = False,
        top_top_interaction: bool = False,
        **kwargs,
    ):
        """
        A single TopoGraphLayer which passes messages between all different types of
        nodes.

        Args
        ----
            edge_net_architecture:
                Architecture to process edges. A separate network is built for each type
                of edge (e.g. jet-W and W-jet are different edges). All edge networks
                have the same HPs.
            node_net_architecture:
                Architecture to process nodes after they have been updated with the
                pooled edges. A separate network is built for each type of node. All
                node networks have the same HPs.
            pooling:
                Pooling operation used to pool edges.
            attention_net_architecture:
                Configuration of the attention network if attention pooling is
                requested. A separate network is built for each type
                of edge. All attention networks have the same HPs.
            full_connection_jets:
                Connect all jets with each other, including self connections.
            full_connections_ws:
                Connect all Ws with each other, including self connections.
            full_connections_tops:
                Connect all Tops with each other, including self connections.
            full_connections_w_top:
                Connect all Ws with all Tops. Otherwise only the first W node will
                interact with the first Top node and the second with the second.
            w_w_interaction:
                Let the two W nodes interact with each other.
            top_top_interaction:
                Let the two Top nodes interact with each other.

        """
        super().__init__(**kwargs)

        self.jet_jet_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges=pooling,
            attention_network=attention_net_architecture,
            k_neighbours=15,
            fully_connected=full_connections_jets,
        )
        self.jet_w_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges=pooling,
            attention_network=attention_net_architecture,
            fully_connected=True,
        )
        self.jet_top_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges=pooling,
            attention_network=attention_net_architecture,
            fully_connected=True,
        )

        self.dense_jets = FCDenseBlock(node_net_architecture)

        self.w_jet_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges=pooling,
            attention_network=attention_net_architecture,
            fully_connected=True,
        )

        if w_w_interaction:
            self.w_w_block = EdgeBlock(
                dense_edges=edge_net_architecture,
                pooling_edges="avg",
                attention_network=None,
                k_neighbours=1,
                fully_connected=full_connections_ws,
            )
        self.w_top_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges="avg",
            attention_network=None,
            single_connection=not full_connections_w_top,
            fully_connected=full_connections_w_top,
        )

        self.dense_ws = FCDenseBlock(node_net_architecture)

        self.top_jet_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges=pooling,
            attention_network=attention_net_architecture,
            fully_connected=True,
        )
        self.top_w_block = EdgeBlock(
            dense_edges=edge_net_architecture,
            pooling_edges="avg",
            attention_network=None,
            single_connection=not full_connections_w_top,
            fully_connected=full_connections_w_top,
        )
        if top_top_interaction:
            self.top_top_block = EdgeBlock(
                dense_edges=edge_net_architecture,
                pooling_edges="avg",
                attention_network=None,
                k_neighbours=1,
                fully_connected=full_connections_tops,
            )

        self.dense_tops = FCDenseBlock(node_net_architecture)

    def call(self, data: TopoGraphData) -> TopoGraphData:
        """
        Call the layer

        Args
        ----
            data:
                TopoGraphData holding jet nodes, W nodes, top nodes, a mask for
                existing/non-existing jets, and potentially previous edges.

        Returns
        -------
            updated_data:
                Updated TopoGraphData. Everything except the mask is updated.

        """
        updated_data = TopoGraphData(
            jets=None, mask=data.mask, previous_edges=[None] * 9
        )

        (pooled_jet_jet_edges, updated_data.previous_edges[0],) = self.jet_jet_block(
            data.jets, data.jets, data.mask, data.mask, data.previous_edges[0]
        )
        pooled_jet_w_edges, updated_data.previous_edges[1] = self.jet_w_block(
            data.jets, data.nodes_w, data.mask, None, data.previous_edges[1]
        )
        (pooled_jet_top_edges, updated_data.previous_edges[2],) = self.jet_top_block(
            data.jets, data.nodes_top, data.mask, None, data.previous_edges[2]
        )

        updated_jets = tf.concat(
            [data.jets, pooled_jet_jet_edges, pooled_jet_w_edges, pooled_jet_top_edges],
            axis=-1,
        )
        updated_data.jets = pass_with_mask(updated_jets, self.dense_jets, data.mask)

        pooled_w_jet_edges, updated_data.previous_edges[3] = self.w_jet_block(
            data.nodes_w, data.jets, None, data.mask, data.previous_edges[3]
        )
        pooled_w_top_edges, updated_data.previous_edges[5] = self.w_top_block(
            data.nodes_w, data.nodes_top, None, None, data.previous_edges[5]
        )
        if hasattr(self, "w_w_block"):
            pooled_w_w_edges, updated_data.previous_edges[4] = self.w_w_block(
                data.nodes_w, data.nodes_w, None, None, data.previous_edges[4]
            )
            updated_ws = tf.concat(
                [
                    data.nodes_w,
                    pooled_w_jet_edges,
                    pooled_w_w_edges,
                    pooled_w_top_edges,
                ],
                axis=-1,
            )
        else:
            updated_ws = tf.concat(
                [data.nodes_w, pooled_w_jet_edges, pooled_w_top_edges], axis=-1
            )

        updated_data.nodes_w = self.dense_ws(updated_ws)

        (pooled_top_jet_edges, updated_data.previous_edges[6],) = self.top_jet_block(
            data.nodes_top, data.jets, None, data.mask, data.previous_edges[6]
        )
        pooled_top_w_edges, updated_data.previous_edges[7] = self.top_w_block(
            data.nodes_top, data.nodes_w, None, None, data.previous_edges[7]
        )
        if hasattr(self, "top_top_block"):
            (
                pooled_top_top_edges,
                updated_data.previous_edges[8],
            ) = self.top_top_block(
                data.nodes_top, data.nodes_top, None, None, data.previous_edges[8]
            )
            updated_tops = tf.concat(
                [
                    data.nodes_top,
                    pooled_top_jet_edges,
                    pooled_top_w_edges,
                    pooled_top_top_edges,
                ],
                axis=-1,
            )
        else:
            updated_tops = tf.concat(
                [data.nodes_top, pooled_top_jet_edges, pooled_top_w_edges], axis=-1
            )
        updated_data.nodes_top = self.dense_tops(updated_tops)

        return updated_data
