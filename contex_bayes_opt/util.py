import warnings
import numpy as np

def acq_max(ac, gp, all_discr_actions, context):
    """
    A function to find the maximum of the acquisition function
    We evaluate all possible actions since we consider a discrete set of actions.
    """
    context_action = np.concatenate([np.tile(context, (len(all_discr_actions), 1)), all_discr_actions], axis=1)
    
    ys = ac(context_action, gp=gp)
    x_max = all_discr_actions[ys.argmax()]
    return x_max
    

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, beta_kind='const', beta_const=1):

        self.beta_const = beta_const
        self.beta_val = 1
        self.t = 0
        self.delta = 0.01

        if kind not in ['ucb']:
            err = "The utility function " \
                  "{} has not been implemented.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind
            
        if beta_kind not in ['const', 'theor']:
            err = "The beta function " \
                  "{} has not been implemented, select" \
                  "const or theor".format(beta_kind)
            raise NotImplementedError(err)
        else:
            self.beta_kind = beta_kind


    def update_params(self):
        self.t += 1
        if self.beta_kind == 'const':
            self.beta_val = self.beta_const
        elif self.beta_kind == 'theor':
            self.beta_val = 2 + 300 * self.t**(33/34) * np.log10(self.t) * ( np.log(self.t / self.delta) ** 3)

    def utility(self, x, gp):
        self.update_params()
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.beta_val)

    @staticmethod
    def _ucb(x, gp, beta):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        return mean + beta * std

