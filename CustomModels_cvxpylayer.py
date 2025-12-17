from keras import layers, Model
import tensorflow as tf

import Custom_Layers as cl

try:
    import cvxpy as cp
    from cvxpylayers.tensorflow import CvxpyLayer
except ImportError as exc:  # pragma: no cover - handled at runtime
    cp = None
    CvxpyLayer = None
    _CVXPY_IMPORT_ERROR = exc
else:
    _CVXPY_IMPORT_ERROR = None


class DifferentiableMinVarianceLayer(layers.Layer):
    """Calcule les poids de variance minimale via une couche CVXPY différentiable."""

    def __init__(
        self,
        allow_short_selling: bool = False,
        n_assets: int | None = None,
        epsilon: float = 1e-6,
        name: str = "DiffMinVarWeights",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.allow_short_selling = allow_short_selling
        self.epsilon = float(epsilon)
        self._n_assets_hint = n_assets
        self._layer_cache: dict[int, CvxpyLayer] = {}

    def _build_cvxpy_layer(self, n_assets: int) -> CvxpyLayer:
        if n_assets <= 0:
            raise ValueError("n_assets doit être strictement positif.")
        if n_assets in self._layer_cache:
            return self._layer_cache[n_assets]

        w = cp.Variable(n_assets)
        Sigma_chol = cp.Parameter((n_assets, n_assets))

        constraints = [cp.sum(w) == 1]
        if not self.allow_short_selling:
            constraints.append(w >= 0)

        objective = cp.Minimize(cp.sum_squares(Sigma_chol @ w))
        problem = cp.Problem(objective, constraints)

        layer = CvxpyLayer(problem, parameters=[Sigma_chol], variables=[w])
        self._layer_cache[n_assets] = layer
        return layer

    def _infer_n_assets(self, sigma: tf.Tensor) -> tuple[tf.Tensor, int]:
        n_assets_tensor = tf.shape(sigma)[-1]
        n_assets_static = sigma.shape[-1]
        if n_assets_static is not None:
            return n_assets_tensor, int(n_assets_static)
        if tf.executing_eagerly():
            return n_assets_tensor, int(n_assets_tensor.numpy())
        if self._n_assets_hint is not None:
            return n_assets_tensor, int(self._n_assets_hint)
        raise ValueError(
            "Le nombre d'actifs doit être connu (soit via la dimension statique, "
            "soit via l'argument n_assets) pour utiliser DifferentiableMinVarianceLayer."
        )

    def build(self, input_shape):
        if CvxpyLayer is None or cp is None:
            raise ImportError(
                "cvxpylayers et cvxpy doivent être installés pour utiliser DifferentiableMinVarianceLayer."
            ) from _CVXPY_IMPORT_ERROR

        if len(input_shape) < 2:
            raise ValueError("L'entrée doit être une matrice carrée (…, n, n).")

        last_dim = input_shape[-1]
        penultimate_dim = input_shape[-2]
        if (
            last_dim is not None
            and penultimate_dim is not None
            and last_dim != penultimate_dim
        ):
            raise ValueError("L'entrée doit être une matrice carrée (…, n, n).")

        initial_assets = None
        if last_dim is not None:
            initial_assets = int(last_dim)
        elif self._n_assets_hint is not None:
            initial_assets = int(self._n_assets_hint)

        if initial_assets is not None:
            self._build_cvxpy_layer(initial_assets)

        super().build(input_shape)

    def call(self, sigma):
        sigma = tf.convert_to_tensor(sigma, dtype=self.compute_dtype)

        leading_shape = tf.shape(sigma)[:-2]
        n_assets_tensor, n_assets_value = self._infer_n_assets(sigma)
        sigma_flat = tf.reshape(sigma, (-1, n_assets_tensor, n_assets_tensor))

        sigma_sym = 0.5 * (sigma_flat + tf.linalg.matrix_transpose(sigma_flat))
        jitter = self.epsilon * tf.eye(n_assets_value, dtype=sigma.dtype)
        sigma_sym += jitter

        sigma_chol = tf.linalg.cholesky(sigma_sym)
        sigma_chol64 = tf.cast(sigma_chol, tf.float64)

        cvxpy_layer = self._build_cvxpy_layer(n_assets_value)
        weights_flat, = cvxpy_layer(sigma_chol64)
        weights_flat = tf.cast(weights_flat, sigma.dtype)

        weights = tf.reshape(
            weights_flat, tf.concat([leading_shape, [n_assets_tensor]], axis=0)
        )
        weights = tf.expand_dims(weights, axis=-1)

        static_leading = sigma.shape[:-2]
        if static_leading:
            weights.set_shape(static_leading + (n_assets_value, 1))
        else:
            weights.set_shape((n_assets_value, 1))

        return weights


class SimpleDiffMinVarModel(tf.keras.Model):
    """Wrappe un backbone et la couche CVXPY pour offrir une API Keras standard."""

    def __init__(self, backbone, opt_layer, requested_outputs, backbone_output_keys, name="SimpleModel"):
        super().__init__(name=name)
        self.backbone = backbone
        self.opt_layer = opt_layer
        self.requested_outputs = list(
            requested_outputs) if requested_outputs else ["weights"]
        self.available_outputs = list(
            backbone_output_keys) + ["weights", "variance"]

    def compile(self, *args, run_eagerly=True, **kwargs):
        super().compile(*args, run_eagerly=run_eagerly, **kwargs)

    def _compute_full_outputs(self, inputs, training):
        if not tf.executing_eagerly():
            raise RuntimeError(
                "SimpleDiffMinVarModel doit être exécuté en mode eager. "
                "Veuillez compiler avec run_eagerly=True ou appeler le modèle "
                "dans une boucle personnalisée."
            )
        base_outputs = self.backbone(inputs, training=training)
        sigma = base_outputs["TransformedCovarianceMatrix"]
        weights = self.opt_layer(sigma)
        sigma_w = tf.matmul(sigma, weights)
        variance = tf.matmul(weights, sigma_w, transpose_a=True)
        variance = tf.squeeze(variance, axis=(-1, -2))

        full_outputs = dict(base_outputs)
        full_outputs["weights"] = weights
        full_outputs["variance"] = variance
        return full_outputs, sigma, weights, variance

    def call(self, inputs, training=False):
        full_outputs, _, _, _ = self._compute_full_outputs(
            inputs, training=training)
        outputs = []
        for name in self.requested_outputs:
            if name not in full_outputs:
                raise ValueError(
                    f"Sortie demandée inconnue: '{name}'. Possibles: {list(full_outputs.keys())}")
            outputs.append(full_outputs[name])

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            _, sigma, weights, variance_pred = self._compute_full_outputs(
                x, training=True)
            # Cast labels to match the model dtype (generators often yield float64)
            sigma_true = tf.cast(sigma if y is None else y, dtype=weights.dtype)
            # Utilise la loss fournie à compile (ex: variance_loss_function)
            loss = self.compiled_loss(
                sigma_true,
                weights,
                sample_weight,
                regularization_losses=self.losses,
            )

        grads = tape.gradient(loss, self.backbone.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.backbone.trainable_variables))

        if y is not None:
            self.compiled_metrics.update_state(
                y, weights, sample_weight)

        logs = {
            "loss": loss,
        }
        # for metric in self.metrics:
        #     logs[metric.name] = metric.result()
        return logs

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        _, sigma, weights, variance_pred = self._compute_full_outputs(
            x, training=False)
        sigma_true = tf.cast(sigma if y is None else y, dtype=weights.dtype)
        loss = self.compiled_loss(
            sigma_true,
            weights,
            sample_weight,
            regularization_losses=self.losses,
        )
        sigma_true_w = tf.matmul(sigma_true, weights)
        variance_true = tf.matmul(weights, sigma_true_w, transpose_a=True)
        variance_true = tf.squeeze(variance_true, axis=(-1, -2))
        variance_mean = tf.reduce_mean(variance_true)

        logs = {
            "loss": loss,
            "variance": variance_mean,
            "weight_mean": tf.reduce_mean(weights),
        }
        for metric in self.metrics:
            logs[metric.name] = metric.result()
        return logs


def SimpleModel(
    n_days,
    n_assets,
    hidden_layer_sizes=[8],
    recurrent_layer_sizes=[32],
    post_recurrent_layer_sizes=[],
    direction='bidirectional',
    recurrent_model='GRU',
    normalize_std='inverse',
    lag_transform=True,
    dropout_rate=0.,
    recurrent_dropout_rate=0.,
    dimensional_features=['n_stocks', 'n_days', 'q'],
    outputs=["weights"],
    **kwargs,
):

    allow_short_selling = kwargs.pop("allow_short_selling", False)

    if n_assets is not None and n_assets <= 0:
        raise ValueError("n_assets doit être strictement positif.")

    # Autorise un nombre d'actifs variable lors de l'inférence.
    inputs = layers.Input(shape=(None, None))
    if lag_transform:
        input_transformed = cl.LagTransformLayer(
            warm_start=True, name="lag_transform_layer")(inputs)
    else:
        input_transformed = inputs

    std, mean = cl.StandardDeviationLayer(
        axis=-1, name='SampleSTD', demean=(not lag_transform))(input_transformed)
    returns = (input_transformed - mean) / std
    CovarianceMatrix = cl.CovarianceLayer(
        expand_dims=False, normalize=True, name='SampleCov')(returns)

    EigenValues, EigenVectors = cl.SpectralDecompositionLayer(
        name='SpectralCov')(CovarianceMatrix)

    EigenValues = cl.DimensionAwareLayer(
        features=dimensional_features, name='DimensionAware')([EigenValues, inputs])

    TransformedEigenvalues = cl.DeepRecurrentLayer(
        recurrent_layer_sizes=recurrent_layer_sizes,
        recurrent_model=recurrent_model,
        direction=direction,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout_rate,
        final_hidden_layer_sizes=post_recurrent_layer_sizes,
        normalize='sum',
        name='EigenvalueTransformation',
    )(EigenValues)

    if len(hidden_layer_sizes) > 0:
        TransformedStd = cl.DeepLayer(
            hidden_layer_sizes + [1],
            last_activation='softplus',
            name='STDTansformation',
        )(std)
        #TransformedStd = cl.CustomNormalizationLayer(axis=-2,mode="sum",name='STDInverseNormalization')(TransformedStd)
    else:
        TransformedStd = 1.0 / std

    TransformedCorrelationMatrix = cl.EigenProductLayer(
        scaling_factor='direct',
        name='TransformedCorrComposition',
    )(TransformedEigenvalues, EigenVectors)

    TransformedCovarianceMatrix = TransformedCorrelationMatrix * cl.CovarianceLayer(
        normalize=False,
        name='AddTransformedSTD',
    )(TransformedStd)

    backbone_outputs = {
        'TransformedCovarianceMatrix': TransformedCovarianceMatrix,
        'TransformedCorrelationMatrix': TransformedCorrelationMatrix,
        'TransformedEigenvalues': TransformedEigenvalues,
        'RescaledReturns': input_transformed,
        'EigenVectors': EigenVectors,
        'Std': std,
        'TransformedStd': TransformedStd,
    }

    backbone = Model(inputs=inputs, outputs=backbone_outputs,
                     name="SimpleBackbone")

    opt_layer = DifferentiableMinVarianceLayer(
        allow_short_selling=allow_short_selling,
        n_assets=n_assets,
        name='PtfWeights',
    )

    return SimpleDiffMinVarModel(
        backbone=backbone,
        opt_layer=opt_layer,
        requested_outputs=outputs,
        backbone_output_keys=backbone_outputs.keys(),
    )


def warm_start_simple_model(
    model_kwargs: dict,
    pretrain_data,
    finetune_data,
    optimizer=None,
    loss=None,
    pretrain_epochs: int = 5,
    finetune_epochs: int = 3,
    batch_size: int = 64,
    verbose: int = 1,
    save_path: str | None = None,
):
    """Pré-entraine un SimpleModel puis le réutilise pour un fine-tuning rapide."""

    def _fit(model, data, epochs: int):
        if isinstance(data, tf.data.Dataset):
            return model.fit(data, epochs=epochs, verbose=verbose)
        return model.fit(*data, epochs=epochs, batch_size=batch_size, verbose=verbose)

    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(1e-3)
    if loss is None:
        loss = cl.variance_loss_function

    pretrain_model = SimpleModel(**model_kwargs)
    pretrain_model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
    pretrain_history = _fit(pretrain_model, pretrain_data, pretrain_epochs)

    warm_model = SimpleModel(**model_kwargs)
    warm_model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)
    warm_model.set_weights(pretrain_model.get_weights())
    finetune_history = _fit(warm_model, finetune_data, finetune_epochs)

    if save_path:
        warm_model.save(save_path)

    return warm_model, pretrain_history, finetune_history


def initialize_cvxpy_backbone_from_simple(simple_model: tf.keras.Model, cvxpy_model: SimpleDiffMinVarModel) -> int:
    """Copie les poids communs du modèle Simple vers le backbone du modèle CVXPY."""
    source_layers = {layer.name: layer for layer in simple_model.layers}
    transferred = 0

    for layer in cvxpy_model.backbone.layers:
        src = source_layers.get(layer.name)
        if src is None:
            continue

        src_weights = src.get_weights()
        tgt_weights = layer.get_weights()
        if not src_weights or len(src_weights) != len(tgt_weights):
            continue
        if any(sw.shape != tw.shape for sw, tw in zip(src_weights, tgt_weights)):
            continue

        layer.set_weights(src_weights)
        transferred += 1

    return transferred
