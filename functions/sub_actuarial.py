
import numpy as np
from global_vars import ALPHA, BETA, GAMMA1, GAMMA2

def get_CFs_vectorized(x, alpha = ALPHA, beta = BETA, gamma1 = GAMMA1, gamma2 = GAMMA2):
    '''
        Get the time-series of cash-flows of contract x up to maturity n.

        Inputs:
        -------
            x: contract details in the form of x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]

        Outsputs:
            CF: time-series of length n-1 (n feature in x). Dimension: Equal to no. of potential outcomes, i.e. alive, dead, disabled, ...
    '''

    assert(len(x.shape)>1)

    _, n, t, freq = [x[:,i].reshape((-1,1))  for i in range(4)]
    S,P = x[:, -2].reshape((-1,1)), x[:, -1].reshape((-1,1))


    N_eff = n/freq
    assert(np.max(N_eff)==np.min(N_eff))
    N_eff = int(N_eff[0])
    N_batch = len(x)

    # matrix with no. of steps per row -> individualization with step-size comes next
    steps = np.dot(freq*np.ones((N_batch, 1)),np.arange(int(N_eff)).reshape(1,N_eff))

    # alive: premium is paid, (alpha, beta, gamma1, gamma2) charges occur
    # Note: steps.shape = (N_batch, N_eff); contract features with x[i].shape = (N_batch, 1)
    # -> broadcasting applied
    CF_live = P*freq*(steps<t) - P*alpha*t*(steps==0) - P*freq*beta*(steps<t) - S*freq*gamma1*(steps<t) - S*freq*gamma1*(steps>=t)
    # dead: premium is paid, Sum insured has to be paid, (alpha, beta, gamma1, gamma2) charges occur
    CF_dead = P*freq*(steps<t) -S - P*alpha*t*(steps==0) - P*freq*beta*(steps<t) - S*freq*gamma1*(steps<t) - S*freq*gamma1*(steps>=t)

    return np.stack([CF_live, CF_dead], axis = -1)


def get_CFs(x, alpha = ALPHA, beta = BETA, gamma1 = GAMMA1, gamma2 = GAMMA2):
    '''
        Get the time-series of cash-flows of contract x up to maturity n.
        Note:   In contrast to get_CFs_vectorized, here we apply the operation for all rows in x, e.g. all contracts, simultaneously - despite different effective lengths n*m
                We perform the same computation as in get_CFs_vectorized, but combine it with a mask, to set the CFs of matured CFs to zero

        Inputs:
        -------
            x: contract details in the form of x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]

        Outputs:
        --------
            CF:     zero-padded array of cash-flows with shape (x.shape[0], max_steps, 2)
                    Last dmension: Equal to no. of potential outcomes, e.g. alive, dead, disabled, ...; we currently only work with dead and alive.
    '''

    assert(len(x.shape)>1)

    # recall: x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
    _, n, t, freq = [x[:,i].reshape((-1,1))  for i in range(4)]
    S,P = x[:, -2].reshape((-1,1)), x[:, -1].reshape((-1,1))
    n_batch = len(x)


    n_eff = (n/freq).astype('int') # effective durations, i.e. duration (in years) times sub-annual observations
    iter_max = max(n_eff) # maximum sequence-length for zero-padding
    mask_matured = (n_eff <= np.arange(iter_max))


    # matrix with no. of steps per row -> individualization with step-size comes next
    steps = np.dot(freq*np.ones((n_batch, 1)),np.arange(iter_max).reshape(1,-1))

    # alive: premium is paid, (alpha, beta, gamma1, gamma2) charges occur
    # Note: steps.shape = (N_batch, N_eff); contract features with x[i].shape = (N_batch, 1)
    # -> broadcasting applied
    CF_live = (P*freq*(steps<t) - P*alpha*t*(steps==0) - P*freq*beta*(steps<t) - S*freq*gamma1*(steps<t) - S*freq*gamma1*(steps>=t))*mask_matured
    # dead: premium is paid, Sum insured has to be paid, (alpha, beta, gamma1, gamma2) charges occur
    CF_dead = (P*freq*(steps<t) -S - P*alpha*t*(steps==0) - P*freq*beta*(steps<t) - S*freq*gamma1*(steps<t) - S*freq*gamma1*(steps>=t))*mask_matured

    return np.stack([CF_live, CF_dead], axis = -1)


def predict_contract_vectorized(x, pmodel, discount, age_scale, bool_print_shapes = False):
    '''
        Perform the Markov-computation of Cash-flows, based on probabilites contained in pmodel.
        Note:   new version which addresses issue of overestimating 1/m_q_x
                before: CF_t accidentally weighted by (1/m+t)_p_x and (1/m+t)_q_m
                now: CF_x weighted by (1/m+t)_p_x and (t-1/m)_p_x*1/m_q_(x+t-1/m) 

        Inputs:
        -------
            x:  raw/ non-scaled contracts --> (important: equals lengths "n/freq" (data-prep) and potentially different step-sizes "freq", i.e. discounting of steps varies)
                form : x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
            pmodel: tf model (FFN type), which predicts p_survive and p_die within the next year, given an age
            discount: fixed, annual discount factor
            age_scale:  maximum age, e.g. 121, which will be used to scale the age feature

        Outputs:
        --------
            sum-value of the probability-weighted and discounted cash-flows
    '''
    N_batch = x.shape[0]

    # check if effective lengths of all contracts in the batch are equal
    N_eff = x[:,1]/x[:,3]
    assert(np.max(N_eff)==np.min(N_eff))
    N_eff = int(N_eff[0])
  
    # CF  shape: (N_batch, N_eff, 2)
    CFs = get_CFs_vectorized(x) #.reshape((N_batch, N_eff, 2))
    
    # desired shape: (N_batch, N_eff, 2) -> use np.swapaxes
    probs = np.array([pmodel.predict(np.array([(x[:,0]+k*x[:,3])/age_scale, x[:,3]], ndmin=2).T) for k in np.arange(N_eff)])
    probs = np.swapaxes(probs, axis1 = 0, axis2 = 1)    
    # cum_probs shape: (N_batch, N_eff); path-wise, N_eff-times multiplication of 1-dim transition-probs
    cum_prod = np.cumprod(probs[:,:,0], axis=1)
    cum_init = np.ones(shape = (len(cum_prod), 1))
    cum_prod = np.concatenate([cum_init, cum_prod[:,0:-1]], axis = 1).reshape(N_batch,N_eff,1) # Note: final dim to allow broadcasting

    prob_eff = probs*cum_prod

    # combine CFs at N_eff - times with probabilities cum_prob to reach that state and discount by GAMMA
    # Note: CFs are solely for p_surv, p_die -> slice cum_prob along axis = -1 -> shape: (N_batch, N_eff, 2)
    # Disconting: broadcast of step-dependent discounting
    discount = discount**np.dot(x[:,3].reshape((N_batch, 1)), np.arange(N_eff).reshape(1,-1)).reshape(N_batch,N_eff,1)
    
    # sum weighted&discounted cash-flows for the batch (along axis=0)
    vals = (prob_eff*CFs*discount)

    if bool_print_shapes:
        print('CFs shape: ', CFs.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))
        print('probs shape: ', probs.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))
        print('cum_prod shape: ', cum_prod.shape, ' expected: ({},{}, 1)'.format(N_batch,N_eff))
        print('prob_eff shape: ', prob_eff.shape, ' expected: ({},{}, 2)'.format(N_batch,N_eff))
        print('discounting shape: ', discount.shape, ' expected: ({},{}, 1)'.format(N_batch,N_eff))
        print('vals shape: ', vals.shape, ' expected: ({},{},{})'.format(N_batch, N_eff, 2))

    return vals.reshape((N_batch,-1)).sum(axis=1).reshape((-1,1)), CFs


def predict_rnn_contract_vectorized(x, y, pmodel, discount, bool_print_shapes = False):
    '''
        Perform the Markov-computation of Cash-flows, based on probabilites contained in pmodel.
        Note:   new version which addresses issue of overestimating 1/m_q_x
                before: CF_t accidentally weighted by (1/m+t)_p_x and (1/m+t)_q_m
                now: CF_x weighted by (1/m+t)_p_x and (t-1/m)_p_x*1/m_q_(x+t-1/m) 

        Inputs:
        -------
            x:  scaled contracts, seqence-shape: (N_batch, N_steps, N_features) 
                important: equals lengths "n/freq" (data-prep) and potentially different step-sizes "freq", i.e. discounting of steps varies)
                form : x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
            y:  cash-flows for contract(s) x, sequence-shape: (N_batch, N_steps, 2)
            pmodel: tf model (RNN type), which predicts p_survive and p_die within the next year, given an age

        Outputs:
        --------
            sum-value of the probability-weighted and discounted cash-flows
    '''
    N_batch = x.shape[0]

    # check if effective lengths of all contracts in the batch are equal
    N_eff = x.shape[1]
  
    # CF  shape: (N_batch, N_eff, 2)
    #CFs = get_CFs_vectorized(x) #.reshape((N_batch, N_eff, 2))
    CFs = y
    assert(CFs.shape[1] == N_eff)
    
    # rrn: pmodel.predict(x) -> shape: (N_batch, N_eff, 2)
    probs = pmodel.predict(x[:,:,[0,3]])
    # cum_probs shape: (N_batch, N_eff); path-wise, N_eff-times multiplication of 1-dim transition-probs
    cum_prod = np.cumprod(probs[:,:,0], axis=1)
    cum_init = np.ones(shape = (len(cum_prod), 1))
    cum_prod = np.concatenate([cum_init, cum_prod[:,0:-1]], axis = 1).reshape(N_batch,N_eff,1) # Note: final dim to allow broadcasting

    prob_eff = probs*cum_prod

    # combine CFs at N_eff - times with probabilities cum_prob to reach that state and discount by GAMMA
    # Note: CFs are solely for p_surv, p_die -> slice cum_prob along axis = -1 -> shape: (N_batch, N_eff, 2)
    # Disconting: broadcast of step-dependent discounting
    discount = discount**(x[:,:,3:4]* np.arange(N_eff).reshape(1,-1,1))
    assert(discount.shape == (N_batch,N_eff,1))
    
    # sum weighted&discounted cash-flows for the batch (along axis=0)
    vals = (prob_eff*CFs*discount)

    if bool_print_shapes:
        print('CFs shape: ', CFs.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))
        print('probs shape: ', probs.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))
        print('cum_prod shape: ', cum_prod.shape, ' expected: ({},{}, 1)'.format(N_batch,N_eff))
        print('prob_eff shape: ', prob_eff.shape, ' expected: ({},{}, 2)'.format(N_batch,N_eff))
        print('discounting shape: ', discount.shape, ' expected: ({},{}, 1)'.format(N_batch,N_eff))
        print('vals shape: ', vals.shape, ' expected: ({},{},{})'.format(N_batch, N_eff, 2))

    return vals.reshape((N_batch,-1)).sum(axis=1).reshape((N_batch,1))


def neural_annuity(freq, v, t_iter, y = None, x = None,  model = None):
    '''
    Compute the annuity value ä_{x:t}^{(m)} = \sum_{k=0}^{tm-1}} {}_{k}p_x
    For simplicity, we display the formula in standard, actuarial notation. 
    Note however, the transition probabilities {}_{k}p_x are taken from the neural network and implicitly depend on the input features (i.e. contract) and the past (semi-Markovian)

    Inputs:
    -------

        freq:   frequency of payments, i.e. 1/m
        v:      discount factor
        t_iter: The number of iterations the premium is paid
                Note: the original t value is the time-in-years. t_iter is time-in-year*payments-per-year
        y:      predicted transition probabilities, i.e. model.predict(x)
                -> faster than providing model and x separately
        x:      optional input if y provided; otherwise x and model required
                contract data in the input-format expected by model; 
        model:  neural network, tf.keras.models.Model
                -> optional, if y not provided
           

    Outputs:
    --------
        annuity:    vector of annuity values of shape (len(x), 1)
    '''

    # check whether either y or x and model provided
    if type(y) == type(None):
        assert(type(x) != type(None))
        assert(type(model) != type(None))

    # check shapes
    if len(freq.shape) == 1:
        freq = freq.reshape((-1,1))
    if len(t_iter.shape) == 1:
        t_iter = t_iter.reshape((-1,1))

    # check if proper inputs are provided
    if type(y) == type(None):
        assert(type(model)!=type(None))


    # Note: data x in format (N_batch, time-steps-to-maturity, N_features)
    # we explicitly need to respect the number of iterations for with premiums are paid
    # model.predict(x) with shape (N_batch, N_sequence, 2) with 2 = # of states, i.e. dead or alive 
    if type(y) != type(None):
        p_survive = y[:,:,0]
    else:
        p_survive = model.predict(x)[:,:,0]
    # Note: do not forget to include {}_{0}p_x = 1
    p_survive = np.concatenate([np.ones((p_survive.shape[0],1)), p_survive], axis= -1)

    # Formula: annuity = \sum_{k=0}^{mt-1} 1/m*v^{k/m}*{}_{k/m}p_{a}
    probs_cum = np.cumprod(p_survive, axis = -1)
    N_batch, N_sequence = probs_cum.shape
    discounting = v**((np.zeros((N_batch, N_sequence))+np.arange(N_sequence))*freq.reshape((N_batch,1)))

    summands = freq*probs_cum*discounting
    # cut off sum after t_iter
    mask = (np.zeros((N_batch, N_sequence))+np.arange(N_sequence)) < t_iter
    summands *= mask

    # print('zero-values in mask: ', np.prod(mask.shape)-np.count_nonzero(mask))
    # print('expected zero-values: ', np.prod(mask.shape)-np.sum(t_iter))

    return np.sum(summands, axis = -1).reshape((-1,1))


def neural_premium_zillmerisation(freq, v, t_iter, alpha, beta, y = None, x = None, model = None):
    '''
    Goal: Given a net-premium lump sum P_0, compute the factor for annuitization and zillmerisation of P_0.
    Steps:
        1) Compute the annuity value ä_{x:t}^{(m)} (-> neural_annuity - function)
        2) For Zillmerisation, include cost-related factors for aquisition-, alpha- and beta-charges
        3) Combine steps 1 and 2, i.e. zill_factor = ä_{x:t}^{(m)}(1-beta/m) -alpha*t
            Note: the specifics of the computation depend on the assumptions on the underlying cost structure (alpha, beta, gamma_1, gamma_2)

    Inputs:
    -------
        freq:   frequency of payments, i.e. 1/m
        v:      discount factor
        t_iter: The number of iterations the premium is paid
                Note: the original t value is the time-in-years. t_iter is time-in-year*payments-per-year
        alpha, beta:    cost-factors, hyperparameters in our setting
        y:      predicted transition probabilities, i.e. model.predict(x)
                -> faster than providing model and x separately
        x:      optional input if y provided; otherwise x and model required
                contract data in the input-format expected by model; 
        model:  neural network, tf.keras.models.Model
                -> optional, if y not provided 

    Outputs:
    --------
        zill_factor:    vector of annuitising and zillmerising factors for each contract
                        shape: (N_contracts, 1)
    '''
    # check appropriate shapes, i.e. avoid 1d-arrays
    if len(freq.shape) == 1:
        freq = freq.reshape((-1,1))
    if len(t_iter.shape) == 1:
        t_iter = t_iter.reshape((-1,1))

    # check if proper inputs are provided
    if type(y) == type(None):
        annuity = neural_annuity(freq, v, t_iter, x=x, model=model)
    else:
        annuity = neural_annuity(freq, v, t_iter, y=y)
    
    # include costs into computation
    return annuity*(1-beta)-alpha*t_iter*freq