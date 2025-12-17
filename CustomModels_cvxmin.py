from keras import layers, Model
import tensorflow as tf
import numpy as np

import Custom_Layers as cl
from CVXPY_classique import minimise_variance


class MinVarianceWeightsLayer(layers.Layer):
    """Calcule les poids de variance minimale via minimise_variance sur chaque covariance."""

    def __init__(
        self,
        allow_short_selling: bool = False,
        n_assets: int | None = None,
        name: str = "MinVarianceWeights",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.allow_short_selling = allow_short_selling
        self._n_assets_hint = n_assets

    def _solve_np(self, sigma_np: np.ndarray) -> np.ndarray:
        sigma_np = np.asarray(sigma_np, dtype=np.float64)
        sigma_np = 0.5 * (sigma_np + sigma_np.T)  # symétrise
        weights = minimise_variance(sigma_np, allow_short_selling=self.allow_short_selling)
        return weights.astype(np.float32)

    def call(self, sigma):
        sigma = tf.convert_to_tensor(sigma, dtype=tf.float32)
        original_shape = tf.shape(sigma)
        n_assets_tensor = tf.shape(sigma)[-1]
        sigma_flat = tf.reshape(sigma, (-1, n_assets_tensor, n_assets_tensor))

        asset_dim_hint = self._n_assets_hint
        static_last = sigma.shape[-1]
        if static_last is not None:
            asset_dim_hint = int(static_last)

        output_shape = (
            (asset_dim_hint,) if asset_dim_hint is not None else (None,)
        )
        output_spec = tf.TensorSpec(shape=output_shape, dtype=tf.float32)

        def solve_single(cov):
            return tf.numpy_function(self._solve_np, [cov], tf.float32)

        weights_flat = tf.map_fn(solve_single, sigma_flat, fn_output_signature=output_spec)

        weights = tf.reshape(weights_flat, tf.concat([original_shape[:-2], [n_assets_tensor]], axis=0))
        weights = tf.expand_dims(weights, axis=-1)

        static_leading = sigma.shape[:-2]
        last_dim = asset_dim_hint if asset_dim_hint is not None else None
        if static_leading:
            weights.set_shape(static_leading + (last_dim, 1))
        else:
            weights.set_shape((last_dim, 1))

        return weights

def EigenvalueRecurrentNN_withScaling(n_days,hidden_layer_sizes, recurrent_layer_sizes,  post_recurrent_layer_sizes=[],
                             direction='bidirectional',recurrent_model='LSTM', 
                             activation="leaky_relu", apply_on_correlation=False,
                             dropout_rate=0.,recurrent_dropout_rate=0., dimensional_features=[], target_return=None,
                             outputs=["weights"],**kwargs):
    
    inputs = layers.Input(shape=(None, n_days))
    
    # If we have a target return, then returns must be scaled by gamma_t
    rescaled_returns,alphas,betas,gammas = cl.Return_Weightings(name='ReturnRescaling', demean=(target_return!=None))(inputs)
        
    if apply_on_correlation==True:
        # If we do not have a target return, then returns must be demeaned, otherwise mean is a zero vector
        std, mean = cl.StandardDeviationLayer(axis=-1,demean=(target_return==None), name='SampleSTD')(rescaled_returns)
        returns = (rescaled_returns-mean)/std
    else:
        returns = rescaled_returns

    CovarianceMatrix = cl.CovarianceLayer(expand_dims=False,normalize=True,name='SampleCov')(returns) 

    EigenValues,EigenVectors = cl.SpectralDecompositionLayer(name='SpectralCov')(CovarianceMatrix)

    EigenValues = cl.DimensionAwareLayer(features=dimensional_features,name='DimensionAware')([EigenValues, inputs])

    TransformedEigenvalues = cl.DeepRecurrentLayer(recurrent_layer_sizes=recurrent_layer_sizes,
                                                    direction=direction,recurrent_model=recurrent_model,
                                                    dropout=dropout_rate,
                                                    recurrent_dropout=recurrent_dropout_rate,
                                                    final_hidden_layer_sizes=post_recurrent_layer_sizes,
                                                    normalize='inverse',name='EigenvalueTransformation')(EigenValues)

    if hidden_layer_sizes[0]>0 and apply_on_correlation==False:
        raise ValueError("hidden_layer_sizes can only be used with apply_on_correlation=True.")
    elif hidden_layer_sizes[0]>0:
        TransformedStd = cl.DeepLayer(list(hidden_layer_sizes)+[1],activation=activation,
                            last_activation='softplus',name='STDTansformation')(std)
        TransformedStd = cl.CustomNormalizationLayer(axis=-2,mode='inverse',
                                                        name='STDInverseNormalization')(TransformedStd)
    else:
        std = cl.CustomNormalizationLayer(axis=-2,mode='sum',name='STDNormalization')(std)
        TransformedStd = 1/std
        
            
    TransformedCorrelationMatrix = cl.EigenProductLayer(scaling_factor='inverse',name='TransformedCorrComposition')(TransformedEigenvalues,EigenVectors) 


    if apply_on_correlation==True:
        TransformedCovarianceMatrix = TransformedCorrelationMatrix * cl.CovarianceLayer(normalize=False,name='AddTransformedSTD')(TransformedStd)
    else:
        TransformedCovarianceMatrix = TransformedCorrelationMatrix

    weights = cl.NormalizedSum(axis_1=-1, axis_2=-2, name='PtfWeights')(TransformedCovarianceMatrix)

    outputs_dict = {
        'weights': weights,
        'TransformedCovarianceMatrix': TransformedCovarianceMatrix,
        'TransformedCorrelationMatrix': TransformedCorrelationMatrix,
        'TransformedEigenvalues': TransformedEigenvalues,
        'RescaledReturns': rescaled_returns,
        'EigenVectors': EigenVectors,
        'TransformedStd': TransformedStd,
        'alphas': alphas,
        'betas': betas,
        'gammas': gammas,
    }

    selected_outputs = [outputs_dict[name] for name in outputs]
    if len(selected_outputs) == 1:
        model = Model(inputs=inputs, outputs=selected_outputs[0])
    else:
        model = Model(inputs=inputs, outputs=selected_outputs)

    return model


def Best_Cov(n_days,hidden_layer_sizes, recurrent_layer_sizes,  post_recurrent_layer_sizes=[],
                             direction='bidirectional',recurrent_model='LSTM', 
                             activation="leaky_relu", apply_on_correlation=False,
                             dropout_rate=0.,recurrent_dropout_rate=0., dimensional_features=[], target_return=None,
                             outputs=["TransformedCovarianceMatrix"],**kwargs):
    
    inputs = layers.Input(shape=(None, n_days))
    
    # If we have a target return, then returns must be scaled by gamma_t
    rescaled_returns,alphas,betas,gammas = cl.Return_Weightings(name='ReturnRescaling', demean=(target_return!=None))(inputs)
        
    if apply_on_correlation==True:
        # If we do not have a target return, then returns must be demeaned, otherwise mean is a zero vector
        std, mean = cl.StandardDeviationLayer(axis=-1,demean=(target_return==None), name='SampleSTD')(rescaled_returns)
        returns = (rescaled_returns-mean)/std
    else:
        returns = rescaled_returns

    CovarianceMatrix = cl.CovarianceLayer(expand_dims=False,normalize=True,name='SampleCov')(returns) 

    EigenValues,EigenVectors = cl.SpectralDecompositionLayer(name='SpectralCov')(CovarianceMatrix)

    EigenValues = cl.DimensionAwareLayer(features=dimensional_features,name='DimensionAware')([EigenValues, inputs])
    
    if apply_on_correlation==True:
        normalize = 'sum'
    else:
        normalize = None

    TransformedEigenvalues = cl.DeepRecurrentLayer(recurrent_layer_sizes=recurrent_layer_sizes,
                                                    direction=direction,recurrent_model=recurrent_model,
                                                    dropout=dropout_rate,
                                                    recurrent_dropout=recurrent_dropout_rate,
                                                    final_hidden_layer_sizes=post_recurrent_layer_sizes,
                                                    normalize=normalize,name='EigenvalueTransformation')(EigenValues)

    if hidden_layer_sizes[0]>0 and apply_on_correlation==False:
        raise ValueError("hidden_layer_sizes can only be used with apply_on_correlation=True.")
    elif hidden_layer_sizes[0]>0:
        TransformedStd = cl.DeepLayer(list(hidden_layer_sizes)+[1],activation=activation,
                            last_activation='softplus',name='STDTansformation')(std)
    elif apply_on_correlation==True:
        std = cl.CustomNormalizationLayer(axis=-2,mode='sum',name='STDNormalization')(std)
        TransformedStd = 1/std
    else:
        TransformedStd = None
        
    if apply_on_correlation==True:
        scaling_factor = 'direct'
    else:
        scaling_factor = 'none'
    TransformedCorrelationMatrix = cl.EigenProductLayer(scaling_factor=scaling_factor,name='TransformedCorrComposition')(TransformedEigenvalues,EigenVectors) 


    if apply_on_correlation==True:
        TransformedCovarianceMatrix = TransformedCorrelationMatrix * cl.CovarianceLayer(normalize=False,name='AddTransformedSTD')(TransformedStd)
    else:
        TransformedCovarianceMatrix = TransformedCorrelationMatrix


    outputs_dict = {

        'TransformedCovarianceMatrix': TransformedCovarianceMatrix,
        'TransformedCorrelationMatrix': TransformedCorrelationMatrix,
        'TransformedEigenvalues': TransformedEigenvalues,
        'RescaledReturns': rescaled_returns,
        'EigenVectors': EigenVectors,
        'TransformedStd': TransformedStd,
        'alphas': alphas,
        'betas': betas,
        'gammas': gammas,
    }

    selected_outputs = [outputs_dict[name] for name in outputs]
    if len(selected_outputs) == 1:
        model = Model(inputs=inputs, outputs=selected_outputs[0])
    else:
        model = Model(inputs=inputs, outputs=selected_outputs)

    return model


def SimpleModel(n_days,
                n_assets=None,
                hidden_layer_sizes=[8],
                recurrent_layer_sizes=[32],
                post_recurrent_layer_sizes=[],
                direction='bidirectional',
                recurrent_model='GRU',
                normalize_std='inverse',
                lag_transform=True,
                dropout_rate=0.,
                recurrent_dropout_rate=0.,
                dimensional_features=['n_stocks','n_days','q'],
                outputs=["weights"],
                **kwargs):
    
    allow_short_selling = kwargs.pop("allow_short_selling", False)

    if n_assets is not None and n_assets <= 0:
        raise ValueError("n_assets doit être strictement positif.")

    inputs = layers.Input(shape=(None, None))
    if lag_transform:
        input_transformed = cl.LagTransformLayer(warm_start=True, name="lag_transform_layer")(inputs)
    else:
        input_transformed = inputs
    std, mean = cl.StandardDeviationLayer(axis=-1, name='SampleSTD', demean=(not lag_transform))(input_transformed)
    returns = (input_transformed-mean)/std
    CovarianceMatrix = cl.CovarianceLayer(expand_dims=False,normalize=True,name='SampleCov')(returns) 

    EigenValues,EigenVectors = cl.SpectralDecompositionLayer(name='SpectralCov')(CovarianceMatrix)

    EigenValues = cl.DimensionAwareLayer(features=dimensional_features,name='DimensionAware')([EigenValues, inputs]) #rsqrt_n_days


    TransformedEigenvalues = cl.DeepRecurrentLayer(recurrent_layer_sizes=recurrent_layer_sizes,
                                                   recurrent_model=recurrent_model,
                                                    direction=direction,
                                                    dropout=dropout_rate,
                                                    recurrent_dropout=recurrent_dropout_rate,
                                                    final_hidden_layer_sizes=post_recurrent_layer_sizes,
                                                   normalize='sum',name='EigenvalueTransformation')(EigenValues)
    if len(hidden_layer_sizes)>0:
        TransformedStd = cl.DeepLayer(hidden_layer_sizes+[1],
                            last_activation='softplus',name='STDTansformation')(std)
        #TransformedStd = cl.CustomNormalizationLayer(axis=-2,mode=normalize_std,name='STDInverseNormalization')(TransformedStd)
    else:
        TransformedStd = 1./std
        
    TransformedCorrelationMatrix = cl.EigenProductLayer(scaling_factor='direct',name='TransformedCorrComposition')(TransformedEigenvalues,EigenVectors) 

    TransformedCovarianceMatrix = TransformedCorrelationMatrix * cl.CovarianceLayer(normalize=False,name='AddTransformedSTD')(TransformedStd)

    weights = MinVarianceWeightsLayer(
        allow_short_selling=allow_short_selling,
        n_assets=n_assets,
        name='PtfWeights'
    )(TransformedCovarianceMatrix)
    
    outputs_dict = {
        'weights': weights,
        'TransformedCovarianceMatrix': TransformedCovarianceMatrix,
        'TransformedCorrelationMatrix': TransformedCorrelationMatrix,
        'TransformedEigenvalues': TransformedEigenvalues,
        'RescaledReturns': input_transformed,
        'EigenVectors': EigenVectors,
        'Std': std,
        'TransformedStd': TransformedStd,
    }

    selected_outputs = [outputs_dict[name] for name in outputs]
    if len(selected_outputs) == 1:
        model = Model(inputs=inputs, outputs=selected_outputs[0])
    else:
        model = Model(inputs=inputs, outputs=selected_outputs)

    return model
