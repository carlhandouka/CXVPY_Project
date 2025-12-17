import tensorflow as tf
from keras import backend as K
from keras import layers, initializers
from typing import Sequence

@tf.keras.utils.register_keras_serializable()
def variance_loss_function(Sigmas_true, weights_predicted, penalty=0.):
    """
    Custom loss function to calculate portfolio variance normalized by the system size.

    Inputs:
    - Sigmas_true: tf.Tensor, true covariance matrices
    - weights_predicted: tf.Tensor, predicted weights

    Outputs:
    - tf.Tensor, portfolio variance
    """
    if penalty > 0:
        n = tf.cast(tf.shape(Sigmas_true)[-1], dtype=tf.float32)
    else:
        n = 1.0

    portfolio_variance = n * tf.matmul(weights_predicted, tf.matmul(Sigmas_true, weights_predicted), transpose_a=True)

    if penalty > 0:
        gross_leverage = tf.reduce_sum(tf.abs(weights_predicted), axis=-2, keepdims=True)
        excess_leverage = gross_leverage - 1
        portfolio_variance +=  penalty * tf.reduce_mean(tf.square(excess_leverage))

    return portfolio_variance

@tf.keras.utils.register_keras_serializable()
def buy_and_hold_volatility_loss_function(ret_out, weights_predicted):
    """
    Custom loss function to calculate the volatility of a buy-and-hold strategy.

    Inputs:
    - ret_out: tf.Tensor, output returns
    - weights_predicted: tf.Tensor, predicted weights

    Outputs:
    - tf.Tensor, buy-and-hold volatility
    """

    n_stocks = tf.shape(ret_out)[1]
        
   # 2) Prezzi relativi cumprod(1+r), con valore iniziale 1
    cum_rel = tf.math.cumprod(1.0 + ret_out, axis=-1)
    price_series = tf.concat([tf.ones_like(cum_rel[..., :1]), cum_rel], axis=-1)

    # 3) Valore del portafoglio buy‑and‑hold
    port_val = tf.reduce_sum(price_series * weights_predicted, axis=1)
    port_val = tf.clip_by_value(port_val, K.epsilon(), tf.float32.max)  # mai 0

    # 4) Log‑returns (più stabili numericamente di r_t = ΔP/P)
    ret = tf.divide(port_val[:, 1:], port_val[:, :-1]) - 1

    # 5) σ campionaria e annualizzazione (assume dati giornalieri)
    vol = tf.math.reduce_std(ret, axis=-1) * tf.sqrt(252.0 * tf.cast(n_stocks, dtype=tf.float32))

    # 6) Loss = media batch (scalar) → ottimo per .compile(loss=…)
    return tf.reduce_mean(vol) 


@tf.keras.utils.register_keras_serializable()
def frobenius_loss_function(Sigmas_true, Sigmas_predicted):
    """ 
    Custom loss function to calculate the Frobenius norm of the difference between predicted and true covariance matrices.
    Inputs:
    - Sigmas_true: tf.Tensor, true covariance matrices
    - Sigmas_predicted: tf.Tensor, predicted covariance matrices
    Outputs:
    - tf.Tensor, Frobenius norm of the difference between predicted and true covariance matrices, normalized by the system size.
    """
    
    n = tf.cast(tf.shape(Sigmas_true)[-1], dtype=tf.float32)

    return tf.sqrt(tf.reduce_sum(tf.square(Sigmas_predicted - Sigmas_true), axis=[-2, -1])) / n

@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='ReturnRescaling'
)
class Return_Weightings(layers.Layer):
    def __init__(self, name=None, demean=False, **kwargs):
        if name is None:
            raise ValueError("Return_Weightings must have a name.")
        self.demean = demean
        super(Return_Weightings, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            name='alpha'
        )
        self.beta = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            name='beta'
        )
        self.gamma = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer=tf.keras.initializers.Zeros(),
            trainable=self.demean,
            name='gamma'
        )

    def call(self, inputs):
        beta_sp = tf.nn.softplus(self.beta)
        weighted_inputs = self.alpha / beta_sp * tf.math.tanh(beta_sp * (inputs - self.gamma))
        return weighted_inputs, self.alpha, beta_sp, self.gamma

    def get_config(self):
        config = super(Return_Weightings, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='StandardDeviationLayer'
)
class StandardDeviationLayer(layers.Layer):
    def __init__(self, axis=1, demean=False, name=None, **kwargs):
        if name is None:
            raise ValueError("StandardDeviationLayer must have a name.")
        super(StandardDeviationLayer, self).__init__(name=name, **kwargs)
        self.axis = axis
        self.demean = demean

    def call(self, x):
        sample_size = tf.cast(tf.shape(x)[self.axis], tf.float32) 
        
        if self.demean:
            mean = tf.reduce_mean(x, axis=self.axis, keepdims=True)
            x = x - mean
            sample_size -= 1
            
        variance = tf.reduce_sum(tf.square(x), axis=self.axis, keepdims=True) / sample_size
        std = tf.sqrt(variance)
        
        if not self.demean:
            mean = tf.zeros_like(std)
            
        return std, mean

    def get_config(self):
        config = super(StandardDeviationLayer, self).get_config()
        config.update({
            'axis': self.axis,
            'demean': self.demean
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='CovarianceLayer'
)
class CovarianceLayer(layers.Layer):
    def __init__(self, expand_dims=False, normalize=True, name=None, **kwargs):
        if name is None:
            raise ValueError("CovarianceLayer must have a name.")
        super(CovarianceLayer, self).__init__(name=name, **kwargs)
        self.expand_dims = expand_dims
        self.normalize = normalize

    def call(self, Returns):
        if self.normalize:
            sample_size = tf.cast(tf.shape(Returns)[-1], tf.float32) 
            Covariance = tf.matmul(Returns, Returns, transpose_b=True) / sample_size
        else:
            Covariance = tf.matmul(Returns, Returns, transpose_b=True)
        if self.expand_dims:
            Covariance = tf.expand_dims(Covariance, axis=-3)
        return Covariance

    def get_config(self):
        config = super(CovarianceLayer, self).get_config()
        config.update({
            'expand_dims': self.expand_dims,
            'normalize': self.normalize
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='SpectralDecompositionLayer'
)
class SpectralDecompositionLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        if name is None:
            raise ValueError("SpectralDecompositionLayer must have a name.")
        super(SpectralDecompositionLayer, self).__init__(name=name, **kwargs)

    def call(self, CovarianceMatrix):
        eigenvalues, eigenvectors = tf.linalg.eigh(CovarianceMatrix)
        return tf.expand_dims(eigenvalues, axis=-1), eigenvectors

    def get_config(self):
        config = super(SpectralDecompositionLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='DimensionAwareLayer'
)
class DimensionAwareLayer(layers.Layer):
    def __init__(self, features, name=None, **kwargs):
        if name is None:
            raise ValueError("DimensionAwareLayer must have a name.")
        super(DimensionAwareLayer, self).__init__(name=name, **kwargs)
        self.features = features

    def _set_attribute(self, value, shape):
        value = tf.expand_dims(value, axis=-1)
        value = tf.broadcast_to(value, shape)
        return value

    def call(self, inputs):
        eigen_values, original_inputs = inputs
        n_stocks = tf.cast(tf.shape(original_inputs)[1], tf.float32)
        n_days = tf.cast(tf.shape(original_inputs)[2], tf.float32)
        final_shape = tf.shape(eigen_values)
        tensors_to_concat = [eigen_values]
        if 'q' in self.features:
            q = n_days / n_stocks
            tensors_to_concat.append(self._set_attribute(q, final_shape))
        if 'n_stocks' in self.features:
            tensors_to_concat.append(self._set_attribute(tf.math.sqrt(n_stocks), final_shape))
        if 'n_days' in self.features:
            tensors_to_concat.append(self._set_attribute(tf.math.sqrt(n_days), final_shape))
        if 'rsqrt_n_days' in self.features:
            rsqrt_n_days = tf.math.rsqrt(n_days)
            tensors_to_concat.append(self._set_attribute(rsqrt_n_days, final_shape))
        transformed_eigen_values = tf.concat(tensors_to_concat, axis=-1)
        return transformed_eigen_values

    def compute_output_shape(self, input_shape):
        eigen_values_shape, _ = input_shape
        additional_features = len(self.features)
        return eigen_values_shape[:-1] + (eigen_values_shape[-1] + additional_features,)

    def get_config(self):
        config = super(DimensionAwareLayer, self).get_config()
        config.update({
            'features': self.features
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='DeepLayer'
)
class DeepLayer(layers.Layer):
    def __init__(self, hidden_layer_sizes, last_activation="linear",
                 activation="leaky_relu", other_biases=True, last_bias=True,
                 dropout_rate=0., kernel_initializer="glorot_uniform", name=None, **kwargs):
        if name is None:
            raise ValueError("DeepLayer must have a name.")
        super(DeepLayer, self).__init__(name=name, **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.last_activation = last_activation
        self.other_biases = other_biases
        self.last_bias = last_bias
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        self.hidden_layers = []
        self.dropouts = []
        for i, size in enumerate(self.hidden_layer_sizes[:-1]):
            layer_name = f"{self.name}_hidden_{i}"
            dropout_name = f"{self.name}_dropout_{i}"
            dense = layers.Dense(size,
                                 activation=self.activation,
                                 use_bias=self.other_biases,
                                 kernel_initializer=self.kernel_initializer,
                                 name=layer_name)
            dropout = layers.Dropout(self.dropout_rate, name=dropout_name)
            self.hidden_layers.append(dense)
            self.dropouts.append(dropout)

        self.final_dense = layers.Dense(self.hidden_layer_sizes[-1],
                                        use_bias=self.last_bias,
                                        activation=self.last_activation,
                                        kernel_initializer=self.kernel_initializer,
                                        name=f"{self.name}_output")

    def build(self, input_shape):
        for dense, dropout in zip(self.hidden_layers, self.dropouts):
            dense.build(input_shape)
            input_shape = dense.compute_output_shape(input_shape)
            dropout.build(input_shape)
        self.final_dense.build(input_shape)
        super(DeepLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for dense, dropout in zip(self.hidden_layers, self.dropouts):
            x = dense(x)
            x = dropout(x)
        outputs = self.final_dense(x)
        return outputs

    def get_config(self):
        config = super(DeepLayer, self).get_config()
        config.update({
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'last_activation': self.last_activation,
            'other_biases': self.other_biases,
            'last_bias': self.last_bias,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_layer_sizes[-1]
        return tuple(output_shape)

@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='DeepRecurrentLayer'
)
class DeepRecurrentLayer(layers.Layer):
    def __init__(self, recurrent_layer_sizes,final_activation="softplus", final_hidden_layer_sizes=[], final_hidden_activation="leaky_relu",
                 direction='bidirectional', dropout=0.,recurrent_dropout=0.,recurrent_model='LSTM', normalize=None, name=None, **kwargs):
        if name is None:
            raise ValueError("DeepRecurrentLayer must have a name.")
        super(DeepRecurrentLayer, self).__init__(name=name, **kwargs)

        self.recurrent_layer_sizes = recurrent_layer_sizes
        self.final_activation = final_activation
        self.final_hidden_layer_sizes = final_hidden_layer_sizes
        self.final_hidden_activation = final_hidden_activation
        self.direction = direction
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_model = recurrent_model
        if normalize not in [None, 'inverse', "sum"]:
            raise ValueError("normalize must be None, 'inverse', or 'sum'.")
        self.normalize = normalize

        RNN = getattr(layers, recurrent_model)

        self.recurrent_layers = []
        for i, units in enumerate(self.recurrent_layer_sizes):
            layer_name = f"{self.name}_rnn_{i}"
            cell_name = f"{layer_name}_cell"
            if self.direction == 'bidirectional':
                cell = RNN(units=units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                           return_sequences=True, name=cell_name)
                rnn_layer = layers.Bidirectional(cell, name=layer_name)
            elif self.direction == 'forward':
                rnn_layer = RNN(units=units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                                return_sequences=True, name=layer_name)
            elif self.direction == 'backward':
                rnn_layer = RNN(units=units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                                return_sequences=True, go_backwards=True, name=layer_name)
            else:
                raise ValueError("direction must be 'bidirectional', 'forward', or 'backward'.")
            self.recurrent_layers.append(rnn_layer)

        self.final_deep_dense = DeepLayer(final_hidden_layer_sizes+[1], 
                                     activation=final_hidden_activation,
                                     last_activation=final_activation,
                                     dropout_rate=dropout,
                                     name=f"{self.name}_finaldeep")       

    def build(self, input_shape):
        for layer in self.recurrent_layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.final_deep_dense.build(input_shape)
        super(DeepRecurrentLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.recurrent_layers:
            x = layer(x)
        outputs = self.final_deep_dense(x)
        if self.normalize is not None:
            outputs = CustomNormalizationLayer(self.normalize, axis=-2, name=f"{self.name}_norm")(outputs)
        return tf.squeeze(outputs, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'recurrent_layer_sizes':      self.recurrent_layer_sizes,
            'final_activation':           self.final_activation,
            'final_hidden_layer_sizes':   self.final_hidden_layer_sizes,
            'final_hidden_activation':    self.final_hidden_activation,
            'direction':                  self.direction,
            'dropout':                    self.dropout,
            'recurrent_dropout':          self.recurrent_dropout,
            'recurrent_model':            self.recurrent_model,
            'normalize':                  self.normalize
        })
        return config

    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='CustomNormalizationLayer'
)
class CustomNormalizationLayer(layers.Layer):
    def __init__(self, mode='sum', axis=-2, name=None, **kwargs):
        if name is None:
            raise ValueError("CustomNormalizationLayer must have a name.")
        super(CustomNormalizationLayer, self).__init__(name=name, **kwargs)
        self.mode = mode
        self.axis = axis

    def call(self, x):
        n = tf.cast(tf.shape(x)[self.axis], dtype=tf.float32)
        if self.mode == 'sum':
            x = n * x / tf.reduce_sum(x, axis=self.axis, keepdims=True)
        elif self.mode == 'inverse':
            inv = tf.math.reciprocal(x)
            x = n * inv / tf.reduce_sum(inv, axis=self.axis, keepdims=True)
            x = tf.math.reciprocal(x)
        return x

    def get_config(self):
        config = super(CustomNormalizationLayer, self).get_config()
        # usa "mode" (come in __init__), non "normalize"
        config.update({
            'mode': self.mode,
            'axis': self.axis
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='EigenProductLayer'
)
class EigenProductLayer(layers.Layer):
    def __init__(self, scaling_factor='none', name=None, **kwargs):
        # prima chiamo il costruttore base, così la cov_layer verrà registrata
        super(EigenProductLayer, self).__init__(name=name, **kwargs)
        
        if name is None:
            raise ValueError("EigenProductLayer must have a name.")
        if scaling_factor not in ['inverse','direct','none']:
            raise ValueError("scaling_factor must be 'inverse', 'direct', or 'none'")
        
        self.scaling_factor = scaling_factor

    def call(self, eigenvalues, eigenvectors):
        """
        eigenvalues: Tensor of shape [..., n]
        eigenvectors: Tensor of shape [..., n, n]
        """
        # 1) costruiamo P_base = V diag(s) V^T
        if self.scaling_factor == 'inverse':
            # per la precision matrix usiamo 1/λ
            s = tf.math.reciprocal(eigenvalues)
        else:
            # per la covariance matrix usiamo λ
            s = eigenvalues

        # broadcast: ogni colonna k di V viene scalata per s[..., k]
        V_scaled = eigenvectors * tf.expand_dims(s, axis=-2)  # [..., n, n]
        P = tf.matmul(V_scaled, eigenvectors, transpose_b=True)  # [..., n, n]

        if self.scaling_factor == 'none':
            return P

        # 2) per direct: normalizziamo P; per inverse: normalizziamo in modo che inv(P) abbia diag=1
        if self.scaling_factor == 'direct':
            # diag(P) = Σ_k λ_k · V_{ik}^2
            diag_P = tf.linalg.diag_part(P)                  # [..., n]
            inv_sqrt = tf.math.rsqrt(diag_P)                 # [..., n] = 1/√diag
            row = tf.expand_dims(inv_sqrt, axis=-1)          # [..., n, 1]
            col = tf.expand_dims(inv_sqrt, axis=-2)          # [..., 1, n]
            return P * row * col

        else:  # 'inverse'
            # vogliamo che inv(P_norm) = correlation
            # ma inv(P)=Σ, e diag(Σ)=Σ_k λ_k · V_{ik}^2
            # calcoliamo quella diag senza rifare matmul full
            # diag_Sigma = tf.reduce_sum(eigenvectors**2 * eigenvalues[...,None,:], axis=-1)
            diag_Sigma = tf.reduce_sum(
                tf.square(eigenvectors) * tf.expand_dims(eigenvalues, -2),
                 axis=-1
            )                                           # [..., n]
            sqrt_d = tf.sqrt(diag_Sigma)                      # [..., n]
            row = tf.expand_dims(sqrt_d, axis=-1)             # [..., n, 1]
            col = tf.expand_dims(sqrt_d, axis=-2)             # [..., 1, n]
            # moltiplichiamo P (la precision base) per D^{1/2} a sinistra e destra
            return P * row * col

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'scaling_factor': self.scaling_factor})
        return cfg
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='NormalizedSum'
)
class NormalizedSum(layers.Layer):
    def __init__(self, axis_1=-1, axis_2=-2, name=None, **kwargs):
        if name is None:
            raise ValueError("NormalizedSum must have a name.")
        super(NormalizedSum, self).__init__(name=name, **kwargs)
        self.axis_1 = axis_1
        self.axis_2 = axis_2
    def call(self, x):
        w = tf.reduce_sum(x, axis=self.axis_1, keepdims=True)
        return w / tf.reduce_sum(w, axis=self.axis_2, keepdims=True)
    
    def get_config(self):
        config = super(NormalizedSum, self).get_config()
        config.update({
            'axis_1': self.axis_1,
            'axis_2': self.axis_2
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='MDRNN2D'
)
class MDRNN2D(tf.keras.layers.Layer):
    """Fully dynamic Multi‑Directional (4‑way) 2‑D RNN layer.

    Accepts tensors of shape **(B, H, W, C)** where *B, H, W* are **all**
    allowed to be dynamic (``None``) ‑ only the channel dimension ``C`` is
    fixed.  Internally we iterate over the *anti‑diagonals* with
    ``tf.while_loop`` so the graph no longer needs the spatial sizes at
    trace‑time.

    Parameters
    ----------
    units : int
        Hidden size *per* scan direction.
    cell_type : {"lstm", "gru"}, default "lstm"
        Recurrent cell family.
    directions : sequence of {"SE", "SW", "NE", "NW"}, default all four
        Which directions to compute (order controls feature concatenation).
    """

    def __init__(self,
                 units: int,
                 cell_type: str = "lstm",
                 directions: Sequence[str] = ("SE", "SW", "NE", "NW"),
                 **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.cell_type = cell_type.lower()
        if self.cell_type not in {"lstm", "gru"}:
            raise ValueError("cell_type must be 'lstm' or 'gru'.")
        self.directions = tuple(directions)
        for d in self.directions:
            if d not in {"SE", "SW", "NE", "NW"}:
                raise ValueError(f"Unknown direction '{d}'.")
        self._cells: Sequence[tf.keras.layers.Layer] = []  # filled in build()

    # ------------------------------------------------------------------
    # Keras plumbing
    # ------------------------------------------------------------------
    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "cell_type": self.cell_type,
            "directions": self.directions,
        })
        return cfg

    def build(self, input_shape):  # (B, H, W, C) – H/W may be None
        CellCls = (tf.keras.layers.LSTMCell
                   if self.cell_type == "lstm" else tf.keras.layers.GRUCell)
        self._cells = [CellCls(self.units, name=f"mdrnn2d_{d.lower()}")
                       for d in self.directions]
        super().build(input_shape)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _direction_to_flip(dir_: str):
        match dir_:
            case "SE":  # ↘
                return False, False
            case "SW":  # ↙
                return False, True
            case "NE":  # ↗
                return True,  False
            case "NW":  # ↖
                return True,  True
            case _:
                raise ValueError(dir_)

    # ------------------------------------------------------------------
    # Core scan – rewritten with tf.while_loop for dynamic H/W
    # ------------------------------------------------------------------
    def _scan_single(self, x: tf.Tensor, cell: tf.keras.layers.Layer,
                     flip_h: bool, flip_w: bool) -> tf.Tensor:
        """Process one direction and return (B, H, W, units) with dynamic H/W."""
        # Normalise scan direction to ↘ by optional flips
        if flip_h:
            x = tf.reverse(x, axis=[1])
        if flip_w:
            x = tf.reverse(x, axis=[2])

        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        dtype = x.dtype

        shape_ = tf.stack([B, H, W, tf.constant(self.units, dtype=tf.int32)])
        h_grid = tf.zeros(shape_, dtype=dtype)
        if self.cell_type == "lstm":
            c_grid = tf.zeros_like(h_grid)

        # ------------------------------------------------------------------
        # Loop over anti‑diagonals via tf.while_loop
        # ------------------------------------------------------------------
        d0 = tf.constant(0, dtype=tf.int32)
        num_steps = H + W - 1  # dynamic tensor

        # Shape invariants for the dynamic tensors inside the loop
        h_shape_inv = tf.TensorShape([None, None, None, self.units])
        if self.cell_type == "lstm":
            c_shape_inv = h_shape_inv

        def _body_gru(d, h_grid):
            # rows on diagonal d
            i0 = tf.maximum(tf.constant(0, tf.int32), d - W + 1)
            i1 = tf.minimum(H - 1, d)
            rows = tf.range(i0, i1 + 1)                      # (N,)
            cols = d - rows                                  # (N,)
            N = tf.shape(rows)[0]

            batch_ids = tf.repeat(tf.range(B), repeats=N)    # (B*N,)
            rows_rep  = tf.tile(rows, [B])                   # (B*N,)
            cols_rep  = tf.tile(cols, [B])                   # (B*N,)
            idx_flat  = tf.stack([batch_ids, rows_rep, cols_rep], axis=1)
            x_flat = tf.gather_nd(x, idx_flat)               # (B*N, C)

            # Gather helpers
            def _gather(state, di, dj):
                rr = rows_rep + di
                cc = cols_rep + dj
                valid = tf.logical_and(tf.logical_and(rr >= 0, rr < H),
                                       tf.logical_and(cc >= 0, cc < W))
                rr_safe = tf.where(valid, rr, tf.zeros_like(rr))
                cc_safe = tf.where(valid, cc, tf.zeros_like(cc))
                idx_nb = tf.stack([batch_ids, rr_safe, cc_safe], axis=1)
                gathered = tf.gather_nd(state, idx_nb)
                gathered = tf.where(tf.expand_dims(valid, -1), gathered,
                                    tf.zeros_like(gathered))
                return gathered, tf.expand_dims(valid, -1)

            h_up,   v_up   = _gather(h_grid, -1,  0)
            h_left, v_left = _gather(h_grid,  0, -1)
            h_ul,   v_ul   = _gather(h_grid, -1, -1)

            h_sum = h_up + h_left + h_ul
            valid_cnt = (tf.cast(v_up, dtype) + tf.cast(v_left, dtype) +
                          tf.cast(v_ul, dtype))
            valid_cnt = tf.maximum(valid_cnt, 1.)
            h_prev = h_sum / valid_cnt

            new_h, _ = cell(x_flat, [h_prev])                # (B*N, units)
            h_grid = tf.tensor_scatter_nd_update(h_grid, idx_flat, new_h)
            return d + 1, h_grid

        def _body_lstm(d, h_grid, c_grid):
            i0 = tf.maximum(tf.constant(0, tf.int32), d - W + 1)
            i1 = tf.minimum(H - 1, d)
            rows = tf.range(i0, i1 + 1)
            cols = d - rows
            N = tf.shape(rows)[0]

            batch_ids = tf.repeat(tf.range(B), repeats=N)
            rows_rep  = tf.tile(rows, [B])
            cols_rep  = tf.tile(cols, [B])
            idx_flat  = tf.stack([batch_ids, rows_rep, cols_rep], axis=1)
            x_flat = tf.gather_nd(x, idx_flat)

            def _gather(state, di, dj):
                rr = rows_rep + di
                cc = cols_rep + dj
                valid = tf.logical_and(tf.logical_and(rr >= 0, rr < H),
                                       tf.logical_and(cc >= 0, cc < W))
                rr_safe = tf.where(valid, rr, tf.zeros_like(rr))
                cc_safe = tf.where(valid, cc, tf.zeros_like(cc))
                idx_nb = tf.stack([batch_ids, rr_safe, cc_safe], axis=1)
                gathered = tf.gather_nd(state, idx_nb)
                gathered = tf.where(tf.expand_dims(valid, -1), gathered,
                                    tf.zeros_like(gathered))
                return gathered, tf.expand_dims(valid, -1)

            h_up,   v_up   = _gather(h_grid, -1,  0)
            h_left, v_left = _gather(h_grid,  0, -1)
            h_ul,   v_ul   = _gather(h_grid, -1, -1)
            c_up,   _      = _gather(c_grid, -1,  0)
            c_left, _      = _gather(c_grid,  0, -1)
            c_ul,   _      = _gather(c_grid, -1, -1)

            h_sum = h_up + h_left + h_ul
            c_sum = c_up + c_left + c_ul
            valid_cnt = (tf.cast(v_up, dtype) + tf.cast(v_left, dtype) +
                          tf.cast(v_ul, dtype))
            valid_cnt = tf.maximum(valid_cnt, 1.)
            h_prev = h_sum / valid_cnt
            c_prev = c_sum / valid_cnt

            _, [new_h, new_c] = cell(x_flat, [h_prev, c_prev])
            h_grid = tf.tensor_scatter_nd_update(h_grid, idx_flat, new_h)
            c_grid = tf.tensor_scatter_nd_update(c_grid, idx_flat, new_c)
            return d + 1, h_grid, c_grid

        # Run the while‑loop
        if self.cell_type == "gru":
            _, h_grid = tf.while_loop(
                lambda d, *_: tf.less(d, num_steps),
                _body_gru,
                loop_vars=(d0, h_grid),
                shape_invariants=(tf.TensorShape([]), h_shape_inv))
        else:
            _, h_grid, c_grid = tf.while_loop(
                lambda d, *_: tf.less(d, num_steps),
                _body_lstm,
                loop_vars=(d0, h_grid, c_grid),
                shape_invariants=(tf.TensorShape([]), h_shape_inv, c_shape_inv))

        # Undo flips
        if flip_w:
            h_grid = tf.reverse(h_grid, axis=[2])
        if flip_h:
            h_grid = tf.reverse(h_grid, axis=[1])
        return h_grid

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def call(self, inputs, **kwargs):
        outputs = []
        for dir_name, cell in zip(self.directions, self._cells):
            flip_h, flip_w = self._direction_to_flip(dir_name)
            outputs.append(self._scan_single(inputs, cell, flip_h, flip_w))
        return outputs[0] if len(outputs) == 1 else tf.concat(outputs, axis=-1)
    
@tf.keras.utils.register_keras_serializable(
    package='Custom_Layers',
    name='LagTransformLayer'
)
class LagTransformLayer(layers.Layer):
    """Layer that applies a lag transformation to the input tensor.
    
    This layer takes a 3D tensor of shape (batch_size, time_steps, features)
    and applies a lag transformation to it, creating a new tensor with the
    specified number of lags.
    Parameters
    ----------
    lags : int
        The number of lags to apply to the input tensor.
    name : str, optional
        Name of the layer.
    """
    
    def __init__(self, warm_start=True, name=None, eps=None, **kwargs):
        if name is None:
            raise ValueError("LagTransformLayer must have a name.")
     
        
        super().__init__(name=name,**kwargs)
        self.eps        = K.epsilon()
        self.warm_start = warm_start

        # valori presi dalla figura
        self._target = dict(c0=2.8, c1=0.20, c2=0.85, c3=0.50, c4=0.05)

    # ---------- utils ------------------------------------------------------
    def _inv_softplus(self, y: float) -> float:
        """Inverse softplus:  x s.t. softplus(x)=y  (numpy scalar)."""
        return float(tf.math.log(tf.math.expm1(y)))          # np.expm1(y)=exp(y)-1

    def _add_param(self, name, target):
        mean_raw = self._inv_softplus(target - self.eps)

        if self.warm_start:
            init = initializers.Constant(mean_raw)
        else:                                      # ±5 % noise (in raw space)
            init = initializers.RandomNormal(mean_raw, 0.05 * tf.math.abs(mean_raw))

        return self.add_weight(
            shape=(), dtype="float32",
            name=f"raw_{name}",
            initializer=init,
            trainable=True,
        )

    # ---------- build ------------------------------------------------------
    def build(self, input_shape):
        self._raw_c0 = self._add_param("c0", self._target["c0"])
        self._raw_c1 = self._add_param("c1", self._target["c1"])
        self._raw_c2 = self._add_param("c2", self._target["c2"])
        self._raw_c3 = self._add_param("c3", self._target["c3"])
        self._raw_c4 = self._add_param("c4", self._target["c4"])
        super().build(input_shape)

    # ---------- forward ----------------------------------------------------
    def _pos(self, x):                 # softplus(x)+ε  ⇒  (0,∞)
        return tf.nn.softplus(x) + self.eps

    def call(self, R):
        T = tf.shape(R)[-1]                          # lunghezza nel tempo

        # t = [T, T‑1, …, 1]  (dtype uguale a R)
        t = tf.cast(tf.range(1, T + 1), R.dtype)     # (1 … T)
        t = tf.reverse(t, axis=[0])                  # (T … 1)

        # ---------- parametri positivi ----------
        c0 = self._pos(self._raw_c0)
        c1 = self._pos(self._raw_c1)
        c2 = self._pos(self._raw_c2)
        c3 = self._pos(self._raw_c3)
        c4 = self._pos(self._raw_c4)

        # ---------- formule ----------
        alpha = c0 * tf.pow(t, -c1)                  # (T,)
        beta  = c2 - c3 * tf.exp(-c4 * t)            # (T,)

        # ---------- reshape & output -------------
        ndims     = tf.rank(R)
        pad_ones  = tf.ones(ndims - 1, dtype=tf.int32)
        shape_T   = tf.concat([pad_ones, [T]], 0)

        alpha_div_beta = tf.reshape(alpha / (beta + self.eps), shape_T)
        beta           = tf.reshape(beta, shape_T)

        return alpha_div_beta * tf.tanh(beta * R)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'eps': self.eps,
            'warm_start': self.warm_start
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
    
