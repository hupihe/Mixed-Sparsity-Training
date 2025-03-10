from __future__ import annotations
from typing import Dict

import numpy as np
import torch
from torch._C import device
import torch.distributed as dist

from DST.utils_nebd import get_W
import os

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def sparse_set(weights:list[torch.Tensor], sparsity:list[float], linear_layers_mask, set_type:str='uniform', ignore_linear_layers=False, keep_first_layer_dense=True, master_process=True)->list[float]:
    # Compute the sparsity of each layer
    ans = []
    if set_type=='uniform':
        for i, (weight, is_linear) in enumerate(zip(weights, linear_layers_mask)):
            # when using uniform sparsity, the first layer is always 100% dense
            # UNLESS there is only 1 layer
            is_first_layer = i == 0
            if is_first_layer and len(weights) > 1 and False:
                ans.append(0)

            elif is_linear and ignore_linear_layers:
                # if choosing to ignore linear layers, keep them 100% dense
                ans.append(0)

            else:
                ans.append(sparsity)
    elif set_type=='ER':
        is_valid = False
        dense_layers = set()
        # dense_layers.add(0)
        while not is_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for i, w in enumerate(weights):
                n_param = w.numel()
                n_zeros = n_param * sparsity
                n_ones = n_param * (1 - sparsity)

                if i in dense_layers:
                    rhs -= n_zeros
                else:
                    rhs += n_ones
                    raw_probabilities[i] = np.sum(w.shape) / w.numel()
                                                
                    divisor += raw_probabilities[i] * n_param
            if len(dense_layers) == len(weights): raise Exception('Cannot set a proper sparsity')
            epsilon = rhs / divisor
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_valid = False
                for weight_i, weight_raw_prob in raw_probabilities.items():
                    if weight_raw_prob == max_prob:
                        print(f"Sparsity of layer {weight_i} has to be set to 0.") if master_process else None
                        dense_layers.add(weight_i)
            else:
                is_valid = True
        for i in range(len(weights)):
            if i in dense_layers:
                ans.append(0)
            else:
                assert raw_probabilities[i] * epsilon >= 0
                ans.append(1 - raw_probabilities[i] * epsilon)
    else:
        raise Exception('Error')
    total_num = np.sum([torch.numel(w) for w in weights])
    total_zero = 0
    for i, w in enumerate(weights):
        n_param = w.numel()
        n_zeros = n_param * ans[i]
        n_ones = n_param * (1 - ans[i])
        total_zero += n_zeros
    print('Real Sparsity:', total_zero/total_num, '\nSparsity distribution:', ans) if master_process else None
    return ans

class IndexMaskHook:
    def __init__(self, layer:torch.Tensor, scheduler:scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return 'IndexMaskHook'

    @torch.no_grad()
    def __call__(self, grad:torch.Tensor):
        mask = self.scheduler.backward_masks[self.layer]
        
        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad/self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask

def _create_step_wrapper(scheduler: scheduler, optimizer: torch.optim.Optimizer):
    _unwrapped_step = optimizer.step
    def _wrapped_step(**kwargs):
        _unwrapped_step(**kwargs)
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
    optimizer.step = _wrapped_step


class scheduler:

    def __init__(
        self, 
        model, 
        optimizer,
        static_topo=False,
        sparsity=0,
        sparsity_distribution='ER',
        ignore_linear_layers=False,
        T_end=None, 
        delta:int=100, 
        zeta:float=0.3,
        random_grow=False, 
        grad_accumulation_n:int=1, 
        zeta_accum=False, 
        load_masks=None, 
        sparsify_type:str='weight',

        use_mest=False,
        mest_lambda=0.01,
        grow_mix=False,
        grad_rand_ratio=3,
        use_RSensitivity=False
        ):

        ddp = int(os.environ.get('RANK', -1)) != -1
        if ddp:
            ddp_rank = int(os.environ['RANK'])
            self.master_process = ddp_rank == 0
        else:
            self.master_process = True
        self.model = model
        self.optimizer:torch.optim.Optimizer = optimizer

        self.random_grow = random_grow
        self.zeta_accum = zeta_accum

        self.W, self._linear_layers_mask = get_W(model, return_linear_layers_mask=True)

        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)
            
        self.sparsity = sparsity
        self.N = [torch.numel(w) for w in self.W]

        self.pruning_curve = []
        self.growning_curve = []
        self.connection_num = []

        self.sparsity_distribution = sparsity_distribution
        self.static_topo = static_topo
        self.grad_accumulation_n = grad_accumulation_n
        self.ignore_linear_layers = ignore_linear_layers
        self.backward_masks:list[torch.Tensor] = None

        self.S = sparse_set(self.W, sparsity, self._linear_layers_mask, sparsity_distribution, self.ignore_linear_layers, True, self.master_process)
        
        # randomly sparsify model according to S
        if load_masks==None:
            if sparsify_type=='random':
                self.random_sparsify()
            elif sparsify_type=='weight':
                self.weight_sparsify()
            else:
                raise Exception('Error')
        else:
            self.backward_masks = load_masks
        #self.weight_sparsify()

        # scheduler keeps a log of how many times it's called. this is how it does its scheduling
        self.step = 0
        self.dst_steps = 0

        # define the actual schedule
        self.delta_T = delta
        self.zeta = zeta
        self.T_end = T_end

        # cos parameter
        self.start = 0
        self.end = T_end
        self.anealing_coeff = 1

        # define start time and rates of growing operations
        self.maximum_sparsity_topo = False
        # also, register backward hook so sparse elements cannot be recovered during normal training
        self.backward_hook_objects:list[IndexMaskHook] = []
        for i, w in enumerate(self.W):
            if getattr(w, '_has_dst_backward_hook', False):
                print('Layer', i, 'already has been registered to a DST_Scheduler.') if self.master_process else None
                # raise Exception('This model already has been registered to a DynamicScheduler.')
        
            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])
            setattr(w, '_has_dst_backward_hook', True)

        assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta


        #################################
        self.use_mest = use_mest,
        self.mest_lambda = mest_lambda

        self.grow_mix = grow_mix
        self.grad_rand_ratio = grad_rand_ratio

        self.use_RSensitivity = use_RSensitivity
        ##################################

    @torch.no_grad()
    def random_sparsify(self):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        for l, w in enumerate(self.W):
            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n)
            perm = perm[:s]
            flat_mask = torch.ones(n, device=w.device)
            flat_mask[perm] = 0
            mask = torch.reshape(flat_mask, w.shape)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    @torch.no_grad()
    def weight_sparsify(self):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        for l, w in enumerate(self.W):
            n = self.N[l]
            s = int(self.S[l] * n)
            score_drop = torch.abs(w)
            # create drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n-s)
            flat_mask = torch.zeros(n, device=w.device)
            flat_mask[sorted_indices] = 1
            mask = torch.reshape(flat_mask, w.shape)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    @torch.no_grad()
    def reset_momentum(self):#reset the momentum according to the mask
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            param_state = self.optimizer.state[w]
            if 'momentum_buffer' in param_state:
                # mask the momentum matrix
                buf = param_state['momentum_buffer']
                buf *= mask


    @torch.no_grad()
    def apply_mask_to_weights(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            w *= mask


    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            w.grad *= mask

    
    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next dst step is, 
        if it's within `self.grad_accumulation_n` steps, return True.
        """

        if self.step >= self.T_end :
            return False

        steps_til_next_dst_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_dst_step <= self.grad_accumulation_n


    def cosine_annealing(self):
        return self.zeta / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))

    def cosine_annealing_new(self):
        return self.zeta / 2 * (1 + np.cos(((self.step - self.start) * np.pi * self.anealing_coeff) / (self.end - self.start)))

    def __call__(self, target_sparsity):
        if self.step == 0:
            self.step = 0.5
        elif self.step == 0.5:
            self.step = 1
        else:
            self.step += 1
        if (self.step % self.delta_T) == 0 and self.step  < self.T_end: # check schedule
            self._dst_step(target_sparsity)
            self.dst_steps += 1
            return False
        return True

    @torch.no_grad()
    def _dst_step(self, target_sparsity):
        assert target_sparsity >= 0 and target_sparsity < 1
        
        total_pruned_num = 0
        total_grown_num = 0
        total_num = 0

        drop_fraction = self.cosine_annealing_new() if not self.static_topo else 0

        # if distributed these values will be populated
        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        current_num = 0
        current_total = 0

        if len(self.connection_num) > 0 and abs(target_sparsity - self.sparsity) < 0.0001:
            self.target_S = self.S
        else:
            self.target_S = sparse_set(self.W, target_sparsity, self._linear_layers_mask, self.sparsity_distribution, self.ignore_linear_layers, True, self.master_process)

        for l, w in enumerate(self.W):

            n_total = self.N[l]
            current_total += n_total

            current_mask = self.backward_masks[l]
            

            # calculate raw scores
            ########################################################################
            if self.use_mest:
                score_drop = torch.abs(w) + self.mest_lambda * torch.abs(self.backward_hook_objects[l].dense_grad)
                score_weight = (torch.abs(w) + self.mest_lambda * torch.abs(self.backward_hook_objects[l].dense_grad)).view(-1)
            elif self.use_RSensitivity:
                score_drop = torch.abs(w/(self.backward_hook_objects[l].dense_grad+1e-9))
                score_weight = (torch.abs(w/(self.backward_hook_objects[l].dense_grad+1e-9))).view(-1)
            ########################################################################
            else:
                score_drop = torch.abs(w)
                score_weight = torch.abs(w).view(-1)
            

            ########################################################################
            if self.grow_mix:
                score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)
            ########################################################################
            elif not self.random_grow:
                score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)
            else:
                score_grow = torch.rand(self.backward_hook_objects[l].dense_grad.size()).to(w.device)
            # if is distributed, synchronize scores
            if is_dist:
                dist.all_reduce(score_drop)  # get the sum of all drop scores
                score_drop /= world_size     # divide by world size (average the drop scores)

                dist.all_reduce(score_grow)  # get the sum of all grow scores
                score_grow /= world_size     # divide by world size (average the grow scores)

            # calculate drop/grow quantities

            n_ones = torch.sum(current_mask).item()

            # create drop mask
            sorted_score, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            sorted_score = sorted_score[:n_ones]
            sorted_indices_temp = sorted_indices[:n_ones]
            if self.zeta_accum:
                threshold_accum = torch.sum(score_weight)*drop_fraction
                n_prune = 0
                while threshold_accum>0:
                    n_prune+=1
                    threshold_accum-=score_weight[sorted_indices_temp[-n_prune]]
            else:
                n_prune = int(n_ones * drop_fraction)

            # define grow fraction
            n_prune = max(n_prune, int((self.target_S[l] - self.S[l]) * self.N[l]))
            n_grow = n_prune - int((self.target_S[l] - self.S[l]) * self.N[l])

            total_num += n_ones
            n_keep = n_ones - n_prune
            total_grown_num += n_grow

            new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_keep,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
            mask1 = new_values.scatter(0, sorted_indices, new_values)
                
            total_pruned_num += n_prune

            # flatten grow scores
            score_grow = score_grow.view(-1)

            # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
            score_grow_lifted = torch.where(
                                mask1 == 1, 
                                torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                                score_grow)

            # create grow mask
            _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)


            ########################################################################
            if self.grow_mix:
                n_grad_grow = int(n_grow * (self.grad_rand_ratio/(self.grad_rand_ratio + 1)))
                n_rand_grow = n_grow - n_grad_grow
                new_values = torch.where(
                            torch.arange(n_total, device=w.device) < n_grad_grow,
                            torch.ones_like(sorted_indices),
                            torch.zeros_like(sorted_indices))
                mask2 = new_values.scatter(0, sorted_indices, new_values)
                zero_indices = torch.where((mask2==0) & (mask1==0))[0]
                random_zero_indices = torch.randperm(zero_indices.size(0))[:n_rand_grow]
                mask2[zero_indices[random_zero_indices]] = 1.
            ########################################################################
            else:
                new_values = torch.where(
                                torch.arange(n_total, device=w.device) < n_grow,
                                torch.ones_like(sorted_indices),
                                torch.zeros_like(sorted_indices))
            
                mask2 = new_values.scatter(0, sorted_indices, new_values)

            mask2_reshaped = torch.reshape(mask2, current_mask.shape)
            grow_tensor = torch.zeros_like(w)
            
            REINIT_WHEN_SAME = False
            if REINIT_WHEN_SAME:
                raise NotImplementedError()
            else:
                new_connections = ((mask2_reshaped == 1) & (current_mask == 0))

            # update new weights to be initialized as zeros and update the weight tensors
            new_weights = torch.where(new_connections.to(w.device), grow_tensor, w)
            w.data = new_weights

            mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()

            # update the mask
            current_mask.data = mask_combined

            current_num += torch.sum(current_mask).item()

            self.S[l] = 1 - torch.sum(current_mask).item() / self.N[l]

            self.reset_momentum()
            self.apply_mask_to_weights()
            self.apply_mask_to_gradients() 

        if total_num > 0:
            self.pruning_curve.append(total_pruned_num/total_num)
            self.growning_curve.append(total_grown_num/total_num)
        if current_total > 0:
            self.connection_num.append(current_num/current_total)
            self.sparsity = 1 - current_num/current_total

        

    @property
    def state_dict(self):
        return {
            'S': self.S,
            'N': self.N,
            'delta_T': self.delta_T,
            'zeta': self.zeta,
            'T_end': self.T_end,
            'static_topo': self.static_topo,
            'sparsity_distribution': self.sparsity_distribution,
            'grad_accumulation_n': self.grad_accumulation_n,
            'step': self.step,
            'dst_steps': self.dst_steps,
            'backward_masks': self.backward_masks
        }

    def load_state_dict(self, state_dict:Dict):
        for k, v in state_dict.items():
            setattr(self, k, v)