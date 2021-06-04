
import numpy as np
from numba import njit, prange

def get_CFs_vectorized(x, alpha = 0.025, beta = 0.03, gamma1 = 0.001, gamma2 = 0.001):
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

    # matrix with no. of steps per row -> individualizacion with step-size comes next
    steps = np.dot(freq*np.ones((N_batch, 1)),np.arange(int(N_eff)).reshape(1,N_eff))

    # alive: premium is paid, (alpha, beta, gamma1, gamma2) charges occur
    # Note: steps.shape = (N_batch, N_eff); contract features with x[i].shape = (N_batch, 1)
    # -> broadcasting applied
    CF_live = P*freq*(steps<t) - P*alpha*t*(steps==0) - P*freq*beta*(steps<t) - S*freq*gamma1*(steps<t) - S*freq*gamma1*(steps>=t)
    # dead: premium is paid, Sum insured has to be paid, (alpha, beta, gamma1, gamma2) charges occur
    CF_dead = P*freq*(steps<t) -S - P*alpha*t*(steps==0) - P*freq*beta*(steps<t) - S*freq*gamma1*(steps<t) - S*freq*gamma1*(steps>=t)

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




##### legacy code ######

# computation of CFs -> not vectorized
# def get_CFs(x, alpha = 0.025, beta = 0.03, gamma1 = 0.001, gamma2 = 0.001):
#     '''
#         Get the time-series of cash-flows of contract x up to maturity n.

#         Inputs:
#         -------
#             x: contract details in the form of x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]

#         Outsputs:
#             CF: time-series of length n-1 (n feature in x). Dimension: Equal to no. of potential outcomes, i.e. alive, dead, disabled, ...
#     '''

#     _, n, t, freq = x[0:4]
#     S,P = x[-2], x[-1]

#     assert(n==int(n))
#     n = int(n)

#     # alive: premium is paid, (alpha, beta, gamma1, gamma2) charges occur
#     CF_live = P*freq*(np.arange(int(n/freq))<t) - P*alpha*t*(np.arange(int(n/freq))==0) - P*freq*beta*(np.arange(int(n/freq))<t) - S*freq*gamma1*(np.arange(int(n/freq))<t) - S*freq*gamma1*(np.arange(int(n/freq))>=t)
#     # dead: premium is paid, Sum insured has to be paid, (alpha, beta, gamma1, gamma2) charges occur
#     CF_dead = P*freq*(np.arange(int(n/freq))<t) -S - P*alpha*t*(np.arange(int(n/freq))==0) - P*freq*beta*(np.arange(int(n/freq))<t) - S*freq*gamma1*(np.arange(int(n/freq))<t) - S*freq*gamma1*(np.arange(int(n/freq))>=t)

#     return np.stack([CF_live, CF_dead], axis = 0)

#-------------------------------------

# non-vectorized version with erraneous computation
# def predict_contract(x, pmodel, discount):
#     '''
#         Perform the Markov-computation of Cash-flows, based on probabilites contained in pmodel.

#         Inputs:
#         -------
#             x: single contract --> vectorize (?!)
#             pmodel: tf model, which predicts p_survive and p_die within the next year, given an age

#         Outputs:
#         --------
#             mean-value of the probability-weighted and discounted cash-flows
#     '''

#     print('Function is out-dated, compare to vectorized version and its adjusted computation.')
#     raise ValueError

    
#     CFs = get_CFs(x)
#     #print('CFs: ', CFs)
#     probs = [np.stack( [pmodel.predict(np.array([age/T_MAX, x[3]], ndmin=2)).flatten(), [0,1]], axis=1) for age in np.arange(int(x[0]),int(x[0])+int(x[1]), x[3])]
#     #print('probs ', probs)
#     cum_prod=np.cumprod(np.array(probs), axis=0)
#     #print('cumprod: ', cum_prod)
#     weighted_CF = np.array([discount**(i*x[3])*np.dot(cum_prod[i][:,0:1].T, CFs[:,i]) for i in range(len(probs))])
#     #print('weighted_CF: ', weighted_CF)
#     val = weighted_CF.mean()

#     # backtest
#     #print('probs: ', [age for age in np.arange(int(x[0]),int(x[0])+int(x[1]), x[3])])
#     #print('cum prob: ', [cum_prod[i][0,0] for i in range(len(cum_prod))],'\n')
#     #p = np.array([probs[i][0,0] for i in range(len(cum_prod))])
#     #print('probs: ', p)
#     #print('cum prob: ', np.cumproduct(p),'\n')

#     return val

#--------------------------------------
# computation of (discounted & mortality-weighted) values -> erraneous implementation
# def predict_contract_vectorized(x, pmodel, discount):
#     '''
#         Perform the Markov-computation of Cash-flows, based on probabilites contained in pmodel.

#         Inputs:
#         -------
#             x:  raw/ non-scaled contracts --> (important: equals lengths "n/freq" (data-prep) and potentially different step-sizes "freq", i.e. discounting of steps varies)
#                 form : x[['x', 'n', 't', 'ZahlweiseNum','Beginnjahr', 'Beginnmonat',  'GeschlechtNum', 'RauchertypNum', 'Leistung', 'tba']]
#             pmodel: tf model, which predicts p_survive and p_die within the next year, given an age

#         Outputs:
#         --------
#             sum-value of the probability-weighted and discounted cash-flows
#     '''
#     N_batch = x.shape[0]

#     # check if effective lengths of all contracts in the batch are equal
#     N_eff = x[:,1]/x[:,3]
#     assert(np.max(N_eff)==np.min(N_eff))
#     N_eff = int(N_eff[0])
  
#     # CF  shape: (N_batch, N_eff, 2)
#     CFs = get_CFs_vectorized(x) #.reshape((N_batch, N_eff, 2))
#     #print('CFs shape: ', CFs.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,2))
#     #print('CFs :', CFs)
    
#     # pmodel.predict.shape: (N_batch, 2) -> with for loop, shape: (N_batch, 2)
#     #inputs = np.array([(x[:,0]+0*x[:,3])/T_MAX, x[:,3]], ndmin=2).T
#     #print('inputs shape: ', inputs.shape, ' expected: ({},{})'.format(N_batch, 2))
#     #preds = pmodel.predict(np.array([(x[:,0]+0*x[:,3])/T_MAX, x[:,3]], ndmin=2).T)#.reshape((N_batch,2))
#     #print('preds shape: ', preds.shape, ' expected: ({},{})'.format(N_batch,2))

#     # filler for 2x2 transition-matrices with [[p_surv, 0], [p_die, 1]]; shape: (N_batch, 2)
#     filler = np.repeat(np.array([[0],[1]],ndmin=1).T, N_batch, axis=0)
#     #print('filler shape: ', filler.shape, ' expected: ({},{})'.format(N_batch,2))

#     # probs shape: after np.stack: (N_eff, N_batch, 2, 2); Note: np.stack adds a new dimension to the tensor
#     # desired shape: (N_batch, N_eff, 2, 2) -> use np.swapaxes
#     probs = np.array([np.stack( [pmodel.predict(np.array([(x[:,0]+k*x[:,3])/T_MAX, x[:,3]], ndmin=2).T),filler], axis=-1) for k in np.arange(N_eff)])
#     probs = np.swapaxes(probs, axis1 = 0, axis2 = 1)    
#     #print('probs shape: ', probs.shape, ' expected: ({},{},{},{})'.format(N_batch,N_eff,2,2))
#     #print('Probs: ', probs)
#     # cum_probs shape: (N_batch, N_eff, 2, 2); path-wise, N_eff-times multiplication of 2x2 trans.-matrices along N_eff-axis  
#     cum_prod = np.cumprod(probs, axis=1)
#     #print('cum_prod shape: ', cum_prod.shape, ' expected: ({},{},{},{})'.format(N_batch,N_eff,2,2))
#     #print('cum_prod: ', cum_prod)

#     # combine CFs at N_eff - times with probabilities cum_prob to reach that state and discount by GAMMA
#     # Note: CFs are solely for p_surv, p_die -> slice cum_prob along axis = -1 -> shape: (N_batch, N_eff, 2)
#     # Disconting: broadcast of step-dependent discounting
#     discount = discount**np.dot(x[:,3].reshape((N_batch, 1)), np.arange(N_eff).reshape(1,-1)).reshape(N_batch,N_eff, 1)
    
#     #print('discounting shape: ', discount.shape, ' expected: ({},{},{})'.format(N_batch,N_eff,1))
#     #print('discounting: ', discount)
    
#     # sum weighted&discounted cash-flows for the batch (along axis=0)
#     vals = (cum_prod[:,:,:,0]*CFs*discount)
#     #print('vals shape: ', vals.shape, ' expected: ({},{},{})'.format(N_batch, N_eff, 2))
#     #print('vals: ', vals)

#     return vals.reshape((N_batch,-1)).sum(axis=1).reshape((-1,1))
