from contex_bayes_opt import ContextualBayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt


def dummy_environment(context, action):
    return np.sum(np.abs(action - (1 - context))) / len(context) 


discvars = {'a1': np.linspace(0, 1, 100),
            'a2': np.linspace(0, 1, 100)}
action_dim = len(discvars)
contexts = {'c1': '', 'c2': ''}
context_dim = len(contexts)


length_scale = np.ones(context_dim+action_dim)
kernel = WhiteKernel(noise_level=1) + Matern(nu=1.5, length_scale=length_scale)
noise = 1e-6

beta_function='const'
beta_const_val=2.5

optimizer = ContextualBayesianOptimization(all_actions_dict=discvars, contexts=contexts, kernel=kernel)

utility = UtilityFunction(kind="ucb", beta_kind=beta_function, beta_const=beta_const_val)

nIters = 200
for i in range(nIters):

    print(i)
    rand_context = np.random.rand(context_dim)
    context = optimizer.array_to_context(rand_context)
    
    action = optimizer.suggest(context, utility)
    
    vContext = optimizer.context_to_array(context)
    vAction = optimizer.action_to_array(action)
    reward = - dummy_environment(vContext, vAction)

    optimizer.register(context, action, reward)

res = optimizer.res
vReward = []
for i in range(nIters):
    vReward.append(res[i]['reward'])
plt.plot(vReward)




