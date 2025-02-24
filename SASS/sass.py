import torch
import copy
import time

import utils as ut

# SASS
class Sass(torch.optim.Optimizer):
    """Implements the SASS algorithm
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        n_batches_per_epoch (int, recommended):: the number batches in an epoch
        init_step_size (float, optional): initial step size (default: 1)
        theta (float, optional): armijo condition constant (default: 0.2)
        gamma_decr (float, optional): multiplicative factor for decreasing the step-size (default: 0.7)
        gamma_incr (float, optional): multiplicative factor for increasing the step-size (default: 1.25)
        alpha_max(float, optional): an upper bound on the step size (default: 10)
        eps_f(float, optional): value for epsilon_f in the Armijo condition (default: 0)
    """

    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 theta=0.2,
                 gamma_decr=0.7,
                 gamma_incr=1.25,
                 alpha_max=10,
                 eps_f=0.0):

        if not 0.0 <= theta < 1:
            raise ValueError("Invalid theta value: {}".format(theta))
        if not 0.0 <= gamma_decr < 1:
            raise ValueError("Invalid gamma decrease: {}".format(gamma_decr))
        if not gamma_incr > 1.0:
            raise ValueError("Invalid gamma increase: {}".format(gamma_incr))
        if not eps_f >= 0:
            raise ValueError("Invalid epsilon_f: {}".format(eps_f))

        defaults = dict(n_batches_per_epoch=n_batches_per_epoch,
                        init_step_size=init_step_size,
                        theta=theta,
                        gamma_decr=gamma_decr,
                        gamma_incr=gamma_incr,
                        alpha_max=alpha_max,
                        eps_f=eps_f)
        super().__init__(params, defaults)

        self.param_dict = defaults

        self.state['step'] = 0
        self.state['eps_f'] = eps_f
        self.state['step_size'] = init_step_size
        self.state['step_size_vs_nn_passes'] = []
        self.state['grad_norm'] = None

        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['total_nn_passes'] = 0
        self.state['loss_vs_nn_passes'] = [] 

## f ##

    def step(self, closure):
        """Performs a single optimization step.
                Arguments:
                    closure (callable, optional): A closure that reevaluates the model
                        and returns the loss.
                """
        seed = time.time()

        def closure_deterministic():
            with ut.random_seed_torch(int(seed)):
                return closure()

        step_size = self.state['step_size']

        # get loss and compute gradients
        loss = closure_deterministic()
        loss.backward()
        self.state['step_size_vs_nn_passes'].append((self.state['total_nn_passes'], self.state['step_size']))
        self.state['total_nn_passes'] += 3
        self.state['loss_vs_nn_passes'].append((self.state['total_nn_passes'], loss.item()))


        # loop over parameter groups
        for group in self.param_groups:
            params = group["params"]

            # save the current parameters:
            params_current = copy.deepcopy(params)
            grad_current = ut.get_grad_list(params)

            grad_norm = ut.compute_grad_norm(grad_current)
            self.state["grad_norm"] = grad_norm

            # step_size = ut.reset_step(step_size=batch_step_size,
            #                           n_batches_per_epoch=group['n_batches_per_epoch'],
            #                           gamma=group['gamma'],
            #                           reset_option=group['reset_option'],
            #                           init_step_size=group['init_step_size'])

            # only do the check if the gradient norm is big enough
            with torch.no_grad():

                # if grad_norm >= 1e-8:

                # check if condition is satisfied
                found = 0
                # step_size_old = step_size

                # try a prospective step
                ut.try_sgd_update(params, step_size, params_current, grad_current)

                # compute the loss at the next step; no need to compute gradients.
                loss_next = closure_deterministic()
                self.state['n_forwards'] += 1

                # =================================================
                # Line search
                success = ut.check_armijo_conditions_nls(step_size=step_size,
                                                         loss=loss,
                                                         grad_norm=grad_norm,
                                                         loss_next=loss_next,
                                                         theta=group['theta'],
                                                         eps_f=self.state['eps_f'])
                if success:
                    step_size = min(step_size * group['gamma_incr'], group['alpha_max'])
                else:
                    step_size = step_size * group['gamma_decr']
                    for p, p_current in zip(params, params_current):
                        p.data = p_current
                self.state['step_size_vs_nn_passes'].append((self.state['total_nn_passes'], self.state['step_size']))


                # # if line search exceeds max_epochs
                # if found == 0:
                #     ut.try_sgd_update(params, 1e-6, params_current, grad_current)

            # save the new step-size
            self.state['step_size'] = step_size
            self.state['step'] += 1

        return loss

    def set_eps_f(self, eps_f):
        self.state['eps_f'] = eps_f

    def get_param_dict(self):
        self.param_dict['eps_f'] = self.state['eps_f']
        return self.param_dict
