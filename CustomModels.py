from keras import layers, Model
import Custom_Layers as cl

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


def SimpleModel(n_days, hidden_layer_sizes=[8], recurrent_layer_sizes=[32],  post_recurrent_layer_sizes=[],
                             direction='bidirectional',recurrent_model='GRU', normalize_std='inverse', lag_transform=True,
                             dropout_rate=0.,recurrent_dropout_rate=0., dimensional_features=['n_stocks','n_days','q'],
                             outputs=["weights"],**kwargs):
    
    inputs = layers.Input(shape=(None, None))  # Input shape can be adjusted as needed
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

    weights = cl.NormalizedSum(axis_1=-1, axis_2=-2, name='PtfWeights')(TransformedCovarianceMatrix)
    
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