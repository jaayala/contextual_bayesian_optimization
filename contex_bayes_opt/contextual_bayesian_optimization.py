import warnings

from .action_space import ActionSpace
from .util import acq_max

from sklearn.gaussian_process import GaussianProcessRegressor


class ContextualBayesianOptimization():
    def __init__(self, all_actions_dict, contexts, kernel, noise=1e-6, points=[], rewards=[], init_random=3):
        
        self._space = ActionSpace(all_actions_dict, contexts)
        self.init_random = init_random
        
        if len(points) > 0:
            gp_hyp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=noise,
                normalize_y=True,
                n_restarts_optimizer=5)
            
            print('Optimizing kernel hyperparameters....')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp_hyp.fit(points, rewards)
            print('Done!')
            
            opt_hyp = gp_hyp.kernel_.get_params()
            kernel.set_params(**opt_hyp)
            optimizer = None
        else:
            warnings.warn('Kernel hyperparameters will be computed during the optimization.')
            optimizer = 'fmin_l_bfgs_b'
            
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=noise,
            normalize_y=True,
            optimizer=optimizer)

    @property
    def space(self):
        return self._space

    @property
    def res(self):
        return self._space.res()

    def register(self, context, action, reward):
        """Expect observation with known reward"""
        self._space.register(context, action, reward)

    def array_to_context(self, context):
        return self._space.array_to_context(context)
    
    def action_to_array(self, action):
        return self._space.action_to_array(action)

    def context_to_array(self, context):
        return self._space.context_to_array(context)

    def suggest(self, context, utility_function):
        """Most promissing point to probe next"""
        assert len(context) == self._space.context_dim
        context = self._space.context_to_array(context)
        if len(self._space) < self.init_random:
            return self._space.array_to_action(self._space.random_sample())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.context_action, self._space.reward)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            all_discr_actions=self._space._allActions,
            context=context)

        return self._space.array_to_action(suggestion)


