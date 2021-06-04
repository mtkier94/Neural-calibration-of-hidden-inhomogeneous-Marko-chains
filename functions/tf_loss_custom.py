import tensorflow as tf


@tf.function#(experimental_relax_shapes = True)
def compute_loss_test(y_true, y_pred):
    '''
    test functionality of custom syntax for tf-loss-functions.
    '''

    return tf.reduce_mean(y_true*y_pred)



# address retracing as different # of steps will results in different shapes and therefore different tf Graphs
@tf.function(experimental_relax_shapes = True) 
def compute_loss_raw(y_true, y_pred):
    '''
    Custom loss for training of tensorflow-model.
    Note:   new version which addresses issue of overestimating 1/m_q_x
            before: CF_t accidentally weighted by (1/m+t)_p_x and (1/m+t)_q_m
            now: CF_x weighted by (1/m+t)_p_x and (t-1/m)_p_x*1/m_q_(x+t-1/m) 

    Inputs:
    -------
        y_true:     target values, i.e. cash-flows in time-series format for respective contracts
                    shape: (None, steps, 2), where 2 equals the number of states, i.e. alive, dead
        y_pred:     predictions, i.e. survival probs as a time-series of equal length than y_true for respective contracts
                    shape: (None, steps, 2)

    Outputs:
    --------
        loss value: tensorflow scalar, corresponding to the mean (per provided contracts) expected value of cash flows, weighted by surv.-probs
    
    Detail on implementation:
        step 1: compute cumulative product of the 1-step survival probs in y_pred, which represents the probability of surviving up to a specific perid.
        step 2: combine cum. prod. with the 1-step transition probabilities (given the ph survived up to the respective period)
    
    '''
    
    # form cumulative survival probabilities to weight paths (of CFs)
    # cumprod along axis of steps, i.e. axis = 1
    # Note: y_pred.shape: (N_batch, N_eff, 2)
    cum_probs = tf.math.cumprod(y_pred[:,:,0:1], axis = 1)
    # broadcast and drop last entry/ last time-step
    cum_probs = tf.broadcast_to(cum_probs[:,0:-1], shape=tf.shape(y_true[:,0:-1,:]))
    ones = tf.ones(tf.shape(y_true[:,0:1,:]))
    cum_prob_concat = tf.concat([ones, cum_probs], axis=1)
    # prob of reaching time t and subsequently transitioning to states described in y_pred
    prob_eff = cum_prob_concat*y_pred
    # CFs at given times weighted by reaching time and transitioning in the respective state
    # Note: discounting factor a-priori included in CFs, i.e. y_true
    values = prob_eff*y_true

    if False:
        # check shapes for debugging
        # Note: for better debugging remove @tf.function decorator to disable graph mode and enable eager performance
        print('shapes y_true, y_pred: ', tf.shape(y_true), tf.shape(y_pred))        
        print('shapes y_true, y_pred: ', y_true.shape, y_pred.shape)   
        print('cum_probs.shape', tf.shape(cum_probs), ' numpy.shape: ', cum_probs.shape)
        print('ones shape:', tf.shape(ones), ones.shape)
        print('cum_prob_concat.shape', tf.shape(cum_prob_concat), 'numpy.shape: ', cum_prob_concat.shape)
        print('prob_eff: ', tf.shape(prob_eff), prob_eff.shape)
        print('shape of values ', tf.shape(values), ' numpy.shapes: ', values.shape)

    return tf.reduce_mean((tf.reduce_sum(values, axis = [1,2])), axis = 0)

@tf.function(experimental_relax_shapes = True) # relax shapes, as we will want to consider different sequence lengths, i.e. different durations of insurance-contracts
def compute_loss_mae(y_true, y_pred):
    '''
    Custom loss for training of tensorflow-model. 

    Inputs:
    -------
        y_true:     target values, i.e. discounted cash-flows in time-series format for respective contracts
                    shape: (None, steps, 2), where 2 equals the number of states, i.e. alive, dead
        y_pred:     predictions, i.e. survival probs as a time-series of equal length than y_true for respective contracts
                    shape: (None, steps, 2)

    Outputs:
    --------
        loss value: tensorflow scalar, corresponding to the mean (per provided contracts) absolute value of cash flows, weighted by predicted surv.-probs
    
    Detail on implementation:
        step 1: compute cumulative product of the 1-step survival probs in y_pred, which represents the probability of surviving up to a specific perid.
        step 2: combine cum. prod. with the 1-step transition probabilities (given the ph survived up to the respective period)
    
    '''
    
    # form cumulative survival probabilities to weight paths (of CFs)
    # cumprod along axis of steps, i.e. axis = 1
    # Note: y_pred.shape: (N_batch, N_eff, 2)
    cum_probs = tf.math.cumprod(y_pred[:,:,0:1], axis = 1)
    # broadcast and drop last entry/ last time-step
    cum_probs = tf.broadcast_to(cum_probs[:,0:-1], shape=tf.shape(y_true[:,0:-1,:]))
    ones = tf.ones(tf.shape(y_true[:,0:1,:]), dtype= y_true.dtype) # Note: dtype might run into compatability-issues depending on the dtype of input (x,y); dtypes should match, as there is no automatic type-conversion
    cum_prob_concat = tf.concat([ones, cum_probs], axis=1)
    # prob of reaching time t and subsequently transitioning to states described in y_pred
    prob_eff = cum_prob_concat*y_pred
    # CFs at given times weighted by reaching time and transitioning in the respective state
    # Note: discounting factor a-priori included in CFs, i.e. y_true
    values = prob_eff*y_true

    if False:
        # check shapes for debugging
        # Note: for better debugging remove @tf.function decorator to disable graph mode and enable eager performance
        print('shapes y_true, y_pred: ', tf.shape(y_true), tf.shape(y_pred))        
        print('shapes y_true, y_pred: ', y_true.shape, y_pred.shape)   
        print('cum_probs.shape', tf.shape(cum_probs), ' numpy.shape: ', cum_probs.shape)
        print('ones shape:', tf.shape(ones), ones.shape)
        print('cum_prob_concat.shape', tf.shape(cum_prob_concat), 'numpy.shape: ', cum_prob_concat.shape)
        print('prob_eff: ', tf.shape(prob_eff), prob_eff.shape)
        print('shape of values ', tf.shape(values), ' numpy.shapes: ', values.shape)

    if False: # for debuggin only; Note: @tf.function decorator has to be turned off and eager_mode = True has to be activated when compiling the respective tf-model
        print('tf-shape: ', values.get_shape())
        #print('tf-value: ', values.numpy())
        print('reduced shape: ', tf.math.abs((tf.reduce_sum(values, axis = [1,2]))).get_shape())
        print('reduced, abs. tf-value: ', tf.math.abs((tf.reduce_sum(values, axis = [1,2], keepdims=True))))
        print('final value: ', tf.reduce_mean(tf.math.abs((tf.reduce_sum(values, axis = [1,2])))))
        print('tf-loss-intern: ', tf.reduce_mean(tf.math.abs((tf.reduce_sum(values, axis = [1,2]))), axis = 0))
    return tf.reduce_mean(tf.math.abs((tf.reduce_sum(values, axis = [1,2]))), axis = 0)
