import pymc3 as pm
import numpy as np

"""
Imagine someone is taking a computeritzed test where every answer is YES/NO

When the test starts, all the answers are already filled in as NO

You know that at some point, the test taker got up and left (leaving a final string of answers unfinished / set to NO)

What is that point, and how skilled is the test taker?
"""

ground_truth = np.array([True,True,True,True,True,True,True,True,True,True])
obs = np.array([True,False,False,False,False,False,False,False,False,False])
correct =  np.equal(ground_truth,obs)

true_inds = np.where(obs)[0]
last_true_idx = true_inds[-1]
test_len = len(ground_truth)

with pm.Model() as model:
    skill = pm.Beta('skill',1.0,1.0)
    n_correct_obs = np.sum(correct)
    
    endpoint = pm.DiscreteUniform('endpoint',last_true_idx,test_len-1)
    
    for i in range(0,last_true_idx+1):
        pm.Bernoulli('correct_%d' % i,skill,observed=correct[i])
        
    for i in range(last_true_idx+1,test_len):
        after_endpoint = pm.math.gt(i,endpoint)
        prob_correct_if_done = float(ground_truth[i]==False)
        prob_correct = pm.math.where(after_endpoint,prob_correct_if_done,skill)
        pm.Bernoulli('correct%d' % i,prob_correct,observed=correct[i])
    
    trace = pm.sample()
    pm.traceplot(trace)
    print pm.summary(trace)
