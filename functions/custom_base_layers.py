"""
Custom tf layers not specific to graph networks or topographs. Contains layers to define
a simple MLP.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Layer,
    LayerNormalization,
)
from tensorflow.keras.regularizers import l2


class DenseLayer(Layer):
    """
    Layer combining a dense layer, an activation layer, and potentially a dropout and
    a normalisation layer.
    """

    def __init__(
        self,
        dense_kwargs: dict,
        activation: str,
        dropout_kwargs: Optional[dict] = None,
        normalisation: Optional[str] = None,
        **kwargs,
    ):
        """
        Layer combining a dense layer, an activation layer, and potentially a dropout
        and a normalisation layer. Batch or layer normalisation can be used for the
        normalisation layer but not both at the same time. The order of layers is dense,
        activation, dropout, normalisation.

        Args
        ----
            dense_kwargs:
                Dictionary holding the kwargs for the tf.keras.layers.Dense layer.
            activation:
                String defining the activation function.
            dropout_kwargs:
                Dictionary holding the kwargs for the tf.keras.layers.Dropout layer. If
                'None' is given no dropout is applied.
            normalisation:
                Type of normalisation. 'batch' gives batch normalisation, 'layer' layer
                normalisation, and everything else no normalisation at all.

        """
        super().__init__(**kwargs)
        self.dense_layer = Dense(**dense_kwargs)
        self.activation_layer = get_activation(activation)
        if dropout_kwargs is not None:
            self.dropout_layer = Dropout(**dropout_kwargs)
        if normalisation.lower() == "batch":
            self.normalisation_layer = BatchNormalization()
        elif normalisation.lower() == "layer":
            self.normalisation_layer = LayerNormalization()

    def call(self, tns):
        """
        Call the layer

        Args
        ----
            tns:
                Inputs to the dense layer
                Expected shape: (batch_size, *, *, n_input_features)
                                (* don't have to be a dimension)

        Returns
        -------
            tns:
                Output of the dense layer with the same shape as the input apart from
                the last dimension which might change.
                Expected shape: (batch_size, *, *, n_output_features)
                                (* don't have to be a dimension)

        """
        tns = self.dense_layer(tns)
        tns = self.activation_layer(tns)
        if hasattr(self, "normalisation_layer"):
            tns = self.normalisation_layer(tns)
        if hasattr(self, "dropout_layer"):
            tns = self.dropout_layer(tns)

        return tns


class FCDenseBlock(Layer):
    """
    Block of dense layers with regularization, activation, dropout, layer norm,
    and batch norm.
    """

    def __init__(
        self,
        architecture: dict,
        additional_output_layer: Optional[dict] = None,
        **kwargs,
    ):
        """
        Create a block of several 'DenseLayers' which combine an actual tf dense layer
        with potential l2 regularisation, an activation function, potential dropout, and
        potential batch or layer normalisation. Batch and layer normalisation can't be
        applied at the same time.
        An additional single dense layer can be added at the end of the block, to have a
        different activation function for the output, e.g. for classification having
        RELu throughout the block, but sigmoid for the last layer.

        Args
        ----
            architecture:
                Dictionary which defines the dense layers. The keys 'units' and
                'activation' need to be present in the dictionary. If the remaining keys
                are not in the dictionary, whatever they set up will not be used.
                units: List of integers. Each entry creates one dense layer with the
                    given integer being the number of nodes.
                activation: String defining the activation function. For possible
                    activation functions see 'get_activation'.
                regularization: Float defining the L2 regularization added to all dense
                    layers.
                dropout: Float giving the rate of dropout to be used. If set to zero no
                    dropout layers are included.
                batch_norm: Bool defining whether to apply batch normalisation or not
                layer_norm: Bool defining whether to apply batch normalisation or not
            additional_output_layer:
                Dictionary defining the final output layer of the dense block.
                units: Integer giving the number of nodes in the final layer
                activation: String defining the activation function directly in the
                    tf.keras.layers.Dense object.

        """
        super().__init__(**kwargs)
        regularization = architecture.get("regularization", 0)
        self.layer_list = []
        do_dropout = architecture.get("dropout", 0) > 0
        do_batch_norm = architecture.get("batch_norm", False)
        do_layer_norm = architecture.get("layer_norm", False)
        if do_batch_norm and do_layer_norm:
            raise ValueError(
                "Requested to do both batch norm and layer norm. Decide for one"
            )
        for unit in architecture["units"]:
            normalisation = None
            dropout_kwargs = None
            if regularization != 0:
                dense_kwargs = {"units": unit, "kernel_regularizer": l2(regularization)}
            else:
                dense_kwargs = {"units": unit}
            if do_batch_norm:
                normalisation = "batch"
            if do_layer_norm:
                normalisation = "layer"
            if do_dropout:
                dropout_kwargs = {"rate": architecture["dropout"]}
            activation = architecture["activation"]

            self.layer_list.append(
                DenseLayer(dense_kwargs, activation, dropout_kwargs, normalisation)
            )
        self.additional_output_layer = additional_output_layer
        if self.additional_output_layer:
            self.output_layer = Dense(
                additional_output_layer["units"],
                activation=additional_output_layer["activation"],
            )

    def call(self, tns: tf.Tensor) -> tf.Tensor:
        """
        Call the layer

        Args
        ----
            tns:
                Inputs to the MLP
                Expected shape: (batch_size, *, *, n_input_features)
                                (* don't have to be a dimension)

        Returns
        -------
            tns:
                Output of the MLP with the same shape as the input apart from the last
                dimension which might change.
                Expected shape: (batch_size, *, *, n_output_features)
                                (* don't have to be a dimension)

        """
        for layers in self.layer_list:
            tns = layers(tns)

        if self.additional_output_layer:
            tns = self.output_layer(tns)

        return tns


def get_activation(activation: str):
    """
    Obtain an activation function/class implemented in tensorflow.

    Args
    ----
        activation:
            String defining the wanted activation function

    Returns
    -------
        Either a tf function implementing the wanted activation function or a tf class.
        Both can just be called to apply the activation function

    """
    if activation.lower() == "leakyrelu":
        return tf.nn.leaky_relu
    if activation.lower() == "elu":
        return tf.nn.elu
    if activation.lower() == "relu":
        return tf.nn.relu
    if activation.lower() == "gelu":
        return tf.nn.gelu
    if activation.lower() == "silu" or activation.lower() == "swish":
        return tf.nn.silu

    raise ValueError(f"Don't know activation function {activation}")
