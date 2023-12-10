import rpyc
import sys, copy, pprint, datetime, time, math, numpy, torch

try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist

CUDA = torch.cuda.is_available()

InterlayerProfileDict = {
    'Default': None, 
    'APF': {
        'freezing_check_frequency' : 1, # unit for freezing_frequency is 'round'
        'ema_alpha' : 0.99,
        'stable_threshold' : 0.01, # 0.01, # 0.05
        'enable_random_freezing' : True
    },
    'Gaia': {
        'initial_stable_threshold' : 0.01
    },
    'CMFL': {
        'initial_relevance_threshold' : 0.7
    },
}

def tensor_list_2_flatten_tensor(tensor_list):
    flatten_tensor = [tensor.view(-1) for tensor in tensor_list]
    flatten_tensor = torch.cat(flatten_tensor, dim=0)
    return flatten_tensor

def flatten_tensor_2_tensor_list(model, flatten_tensor):
    full_tensor = copy.deepcopy(flatten_tensor)
    tensor_list = []
    for param in model.parameters():
        tensor = torch.tensor(full_tensor[:param.numel()])
        tensor = tensor.view(param.size())
        full_tensor = full_tensor[param.numel():]
        tensor_list.append(tensor)
    return tensor_list


class APF_Interlayer:
    def __init__(self, model, sync_frequency, apf_hyperparams):

        self.interlayer_type = 'APF'
        self.model = model
        self.sync_frequency = sync_frequency

        ''' Set hyper-parameters '''
        self.ema_alpha = apf_hyperparams['ema_alpha']
        self.stable_threshold = apf_hyperparams['stable_threshold'] # this shall be larger than 1-self.ema_alpha
        self.tighter_stable_criterion_threshold = 0.8
        self.enable_random_freezing = apf_hyperparams['enable_random_freezing']

        ''' Initialize statistics variables '''
        self.freezing_check_frequency = apf_hyperparams['freezing_check_frequency']  # the unit is one synchronization round
        self.fc_round_id = 0  # freezing-check_round_id
        self.freezing_ratio = 0

        ''' Initialize model-structure-related variables '''
        self.param_shape_list = []
        self.cutoff_index_list = []
        self.tensor_of_flattened_params = None
        self.last_tensor_of_flattened_params = None
        self.flattened_param_length = -1 # this is indeed "self.total_param_num = sum([p.data.nelement() for p in self.model.parameters()])"
        self.init_model_structure_related_variables()

        ''' Initialize apf-algorithm-related statistics variables '''
        self.active_mask = torch.ones(self.flattened_param_length).byte().cuda() if CUDA else torch.ones(self.flattened_param_length).byte()
        self.active_index = self.active_mask.nonzero()
        self.grad_ema = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        self.abs_grad_ema = torch.zeros(self.flattened_param_length).cuda() if CUDA else torch.zeros(self.flattened_param_length)
        self.freezing_lengths = torch.zeros(self.flattened_param_length).int().cuda() if CUDA else torch.zeros(self.flattened_param_length).int()
        self.fc_round_ids_to_unfreeze_params = torch.zeros(self.flattened_param_length).int().cuda() if CUDA else torch.zeros(self.flattened_param_length).int()

        self.logging('create APF Interlayer', apf_hyperparams)

    def init_model_structure_related_variables(self):
        """ Initialize model-structure-related variables by flatten the model parameters one by one. """
        s_index = 0
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            self.param_shape_list.append(p.data.shape)
            flattened_param = p.data.view(-1)
            e_index = s_index + len(flattened_param)
            self.cutoff_index_list.append([s_index, e_index])
            s_index = e_index
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
        self.flattened_param_length = len(self.tensor_of_flattened_params)
        self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)

    def logging(self, string, hyperparameters=None): # each class shall have a 'logging' function right after initialization function
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [APF Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print
        sys.stdout.flush()

    def update_param_freezing_periods_in_TCP_manner(self):
        """ update the active mask of each parameter based on global gradient stability. """
        ''' Calculate the effective stepping rate of each parameter (in tensor mode for fast processing). '''
        self.active_index = self.active_mask.nonzero()
        self.grad = self.last_tensor_of_flattened_params - self.tensor_of_flattened_params
        self.grad_ema[self.active_index] = self.grad_ema[self.active_index] * self.ema_alpha + self.grad[self.active_index] * (1.0 - self.ema_alpha) 
        self.abs_grad_ema[self.active_index] = self.abs_grad_ema[self.active_index] * self.ema_alpha + torch.abs(self.grad[self.active_index]) * (1.0 - self.ema_alpha)
        effective_stepping_rate = torch.abs(self.grad_ema[self.active_index]) / self.abs_grad_ema[self.active_index]

        ''' Update the freezing period length and also the should-be-unfrozen frequency_check_round_id based on effective stepping rate of each param. '''
        self.freezing_lengths[self.active_index] = torch.where(effective_stepping_rate < self.stable_threshold, self.freezing_lengths[self.active_index]+1, self.freezing_lengths[self.active_index]//2)
        self.fc_round_ids_to_unfreeze_params[self.active_index] = self.fc_round_id + self.freezing_lengths[self.active_index] + 1
        if self.enable_random_freezing:
            self.randomly_freeze_active_params()
        self.last_tensor_of_flattened_params = copy.deepcopy(self.tensor_of_flattened_params)

    def refresh_freezing_status(self):
        """ Called after each global synchronization is conducted. Update the freezing status and related statistics."""
        ''' Update the freezing period in a TCP manner and then update the freezing mask accordingly '''
        self.update_param_freezing_periods_in_TCP_manner()
        self.fc_round_id += 1
        self.active_mask = (self.fc_round_ids_to_unfreeze_params == self.fc_round_id)
        ''' Record the stable-ratio statistics and adaptively tune stability threshold if necessary '''
        self.freezing_ratio = 1 - self.active_mask.sum().float() / self.flattened_param_length
        self.logging('current stable ratio: %.4f' % self.freezing_ratio)
        if self.freezing_ratio > self.tighter_stable_criterion_threshold:
            self.stable_threshold /= 2.0
            self.logging('make stable criterion tighter')

    def randomly_freeze_active_params(self):
        # rand_array = torch.rand(self.active_index.shape) * 100
        # rand_frozen = torch.where(rand_array < 50, torch.ones(rand_array.shape).int() , torch.zeros(rand_array.shape).int())
        # rand_frozen = rand_frozen.cuda() if CUDA else rand_frozen 
        # self.fc_round_ids_to_unfreeze_params[self.active_index] = self.fc_round_ids_to_unfreeze_params[self.active_index] + rand_frozen 
        rand_array = torch.rand(self.flattened_param_length)
        rand_frozen = torch.where(rand_array < 0.5, torch.ones(rand_array.shape).int() , torch.zeros(rand_array.shape).int())
        rand_frozen = rand_frozen.cuda() if CUDA else rand_frozen
        self.fc_round_ids_to_unfreeze_params = self.fc_round_ids_to_unfreeze_params + rand_frozen

    """ Below are the public APIs callable by Sync_Manager. """
    def generate_tensor_list_to_transmit(self, iter_id):
        """ Create a list of tensor (in fact with only one element) to sync. """
        ''' flatten the parameters into a 1-dimension list. '''
        self.tensor_of_flattened_params = torch.tensor([]).cuda() if CUDA else torch.tensor([])
        for p in self.model.parameters():
            flattened_param = p.data.view(-1)
            self.tensor_of_flattened_params = torch.cat((self.tensor_of_flattened_params, flattened_param),0)
        ''' Then roll back those should-be-frozen parameters. '''
        self.tensor_of_flattened_params = torch.where(self.active_mask > 0, self.tensor_of_flattened_params, self.last_tensor_of_flattened_params)
        ''' If no synchronization, directly restore the model parameters and return. '''
        if iter_id % self.sync_frequency != 0:
            for i, p in enumerate(self.model.parameters()):
                p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])
            return []
        ''' Then select those unfrozen parameters and package them into a new tensor for transmission'''
        tensor_to_transmit = torch.masked_select(self.tensor_of_flattened_params, self.active_mask)
        return [tensor_to_transmit]

    def restore_model_from_tensor_list_received(self, tensor_list_received):
        self.tensor_of_flattened_params[self.active_mask] = tensor_list_received[0]  # for APF, there should be only one element in tensor_list_received
        ''' Unflattern parameters to model parameters. '''
        for i, p in enumerate(self.model.parameters()):
            p.data = self.tensor_of_flattened_params[self.cutoff_index_list[i][0]:self.cutoff_index_list[i][1]].view(self.param_shape_list[i])
        ''' Refresh the active mask after one synchronization round finishes '''
        self.refresh_freezing_status()

class Gaia_Interlayer:
    def __init__(self, model, sync_frequency, gaia_hyperparams):
        self.interlayer_type = 'Gaia'
        self.model = model
        self.sync_frequency = sync_frequency
        self.total_param_num = sum([p.data.nelement() for p in self.model.parameters()])
        self.initial_stable_threshold = gaia_hyperparams['initial_stable_threshold']
        self.significant_mask_list = [torch.ones(p.data.shape).byte().cuda() if CUDA else torch.ones(p.data.shape).byte() for p in self.model.parameters()]
        self.significant_ratio = 1.0
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]

        self.logging('create Gaia Interlayer', gaia_hyperparams)

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [Gaia Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print 
        sys.stdout.flush()

    def generate_tensor_list_to_transmit(self, iter_id):
        """ Roll back those locally-stable parameters to their previous values. """
        """ TODO: Current the communication reduction is faked. """
        if iter_id % self.sync_frequency != 0:
            return []
        tensor_list_to_transmit = []
        # stable_threshold = self.initial_stable_threshold / math.sqrt(iter_id)
        stable_threshold = self.initial_stable_threshold / math.sqrt(iter_id/10000.0+1.0)
        for i, param in enumerate(self.model.parameters()):
            self.significant_mask_list[i] = (torch.abs(self.last_param_list[i]/param.data - 1) > stable_threshold)
            tensor_to_transmit = torch.where(self.significant_mask_list[i], param.data, self.last_param_list[i])
            tensor_list_to_transmit.append(tensor_to_transmit)
        # problem-1: no actual compression
        # problem-2: no error accumulation
        self.significant_ratio = sum([float(significant_mask.sum()) for significant_mask in self.significant_mask_list]) / self.total_param_num
        self.logging('current significant ratio: %.4f' % self.significant_ratio)
        return tensor_list_to_transmit

    def restore_model_from_tensor_list_received(self, tensor_list_received):
        for i, param in enumerate(self.model.parameters()):
            param.data = torch.where(self.significant_mask_list[i], tensor_list_received[i], param.data)
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]

class CMFL_Interlayer:
    def __init__(self, model, sync_frequency, cmfl_hyperparams):
        self.interlayer_type = 'CMFL'
        self.model = model
        self.sync_frequency = sync_frequency
        self.total_param_num = sum([float(p.data.nelement()) for p in self.model.parameters()])
        self.initial_relevance_threshold = cmfl_hyperparams['initial_relevance_threshold']
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]
        self.last_global_update_list = [torch.zeros(p.data.shape).cuda() if CUDA else torch.zeros(p.data.shape) for p in self.model.parameters()]
        self.logging('create CMFL Interlayer', cmfl_hyperparams)
    
    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [CMFL Interlayer] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
            print 
        sys.stdout.flush()

    def get_relevance_to_global_update(self):
        relevant_element_number = 0
        for i, param in enumerate(self.model.parameters()):
            relevance_tensor = torch.where((param.data-self.last_param_list[i]) * self.last_global_update_list[i] >= 0, \
                torch.ones(param.data.shape).cuda() if CUDA else torch.ones(param.data.shape), torch.zeros(param.data.shape).cuda() if CUDA else torch.zeros(param.data.shape))
            relevant_element_number += float(relevance_tensor.sum())
        relevance = float(relevant_element_number) / self.total_param_num
        return relevance

    def generate_tensor_list_to_transmit(self, iter_id):
        """ Faked version. Instead of excluding self from all-reduce, packing initial param into tensor_list """
        if iter_id % self.sync_frequency != 0:
            return []
        relevance_threshold = self.initial_relevance_threshold / math.sqrt(iter_id/100000.0+1.0)
        relevance_threshold = self.initial_relevance_threshold
        current_relevance = self.get_relevance_to_global_update()
        self.logging('relevance threshold: %.4f; current relevance: %.4f; shall_report: %s'  % (relevance_threshold, current_relevance, 'True' if current_relevance > relevance_threshold else 'False'))
        if current_relevance > relevance_threshold:
            ''' If relevant to global update, then report local update to parameter server. '''
            tensor_list_to_transmit = [p.data for p in self.model.parameters()]
        else:
            ''' Else report initial parameters to PS. '''
            tensor_list_to_transmit = copy.deepcopy(self.last_param_list)
        return tensor_list_to_transmit

    def restore_model_from_tensor_list_received(self, tensor_list_received):
        self.last_global_update_list = []
        for i, param in enumerate(self.model.parameters()):
            param.data = tensor_list_received[i]
            self.last_global_update_list.append(tensor_list_received[i] - self.last_param_list[i])
        self.last_param_list = [copy.deepcopy(p.data) for p in self.model.parameters()]

class Default_Interlayer:
    def __init__(self, model, sync_frequency):
        self.interlayer_type = 'Default'
        self.model = model
        self.sync_frequency = sync_frequency
        self.logging('create Default Interlayer')
    
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] [Default Interlayer] '+str(string))
        sys.stdout.flush()
    
    def generate_tensor_list_to_transmit(self, iter_id):
        if iter_id % self.sync_frequency != 0:
            return []
        tensor_list_to_transmit = []
        for param in self.model.parameters():
            tensor_list_to_transmit.append(param.data)
        return tensor_list_to_transmit
    
    def restore_model_from_tensor_list_received(self, tensor_list_received):
        for i, param in enumerate(self.model.parameters()):
            # Note that you are actually assigning a variable to itself
            param.data = tensor_list_received[i]

class Sync_Manager:
    """ This object shall be able to run without apf (support standard FL by default) """
    def __init__(self, model,rank, sync_profile):
        self.model = model
        self.rank = rank 
        self.sync_round_id = 0
        self.world_size = -1
        self.sync_frequency = sync_profile['sync_frequency']
        self.transmit_interlayer = self.create_transmit_interlayer(sync_profile['interlayer_type'])
                
        rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
        rpyc_config['allow_pickle'] = True
        rpyc_config['sync_request_timeout'] = 1500
        self.connection = rpyc.connect(sync_profile['server_ip'], sync_profile['server_port'], config=rpyc_config)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Sync Manager] '+str(string))
        sys.stdout.flush()

    def create_transmit_interlayer(self, interlayer_type):
        if interlayer_type == 'Default':
            transmit_interlayer = Default_Interlayer(self.model, self.sync_frequency)
        if interlayer_type == 'APF':
            transmit_interlayer = APF_Interlayer(self.model, self.sync_frequency, InterlayerProfileDict[interlayer_type])
        if interlayer_type == 'Gaia':
            transmit_interlayer = Gaia_Interlayer(self.model, self.sync_frequency, InterlayerProfileDict[interlayer_type])
        if interlayer_type == 'CMFL':
            transmit_interlayer = CMFL_Interlayer(self.model, self.sync_frequency, InterlayerProfileDict[interlayer_type])
        return transmit_interlayer

    def try_sync_model(self, iter_id):
        ''' Conduct remote synchronization. '''
        tensor_list_to_transmit = self.transmit_interlayer.generate_tensor_list_to_transmit(iter_id)
        if tensor_list_to_transmit != []:
            self.sync_round_id += 1
            self.logging('report at iter_id: %d ' % iter_id)
            tensor_list_to_transmit = [i.cpu() for i in tensor_list_to_transmit] if CUDA else tensor_list_to_transmit
            tensor_list_received = self.connection.root.aggregate(self.rank, self.sync_round_id, tensor_list_to_transmit)
            tensor_list_received = rpyc.classic.obtain(tensor_list_received)
            if CUDA:
                tensor_list_received = [tensor.cuda() for tensor in tensor_list_received]
            self.logging("receive global model")
            self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received)
            return True
        return False
      
    def try_sync_model_nova(self, iter_id):
        ''' Conduct remote synchronization. '''
        tensor_list_to_transmit = self.transmit_interlayer.generate_tensor_list_to_transmit(iter_id)
        if tensor_list_to_transmit != []:
            self.sync_round_id += 1
            self.logging('report at iter_id: %d ' % iter_id)
            tensor_list_to_transmit = [i.cpu() for i in tensor_list_to_transmit] if CUDA else tensor_list_to_transmit
            tensor_list_received = self.connection.root.aggregate(self.rank, self.sync_round_id, tensor_list_to_transmit, iter_id)
            tensor_list_received = rpyc.classic.obtain(tensor_list_received)
            if CUDA:
                tensor_list_received = [tensor.cuda() for tensor in tensor_list_received]
            self.logging("receive global model")
            self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received)
            return True
        return False

    def try_sync_model_scaffold(self, iter_id, Delta_C_i):
        ''' Conduct remote synchronization. '''
        tensor_list_to_transmit = self.transmit_interlayer.generate_tensor_list_to_transmit(iter_id)
        if tensor_list_to_transmit != []:
            self.sync_round_id += 1
            self.logging('report at iter_id: %d ' % iter_id)
            tensor_list_to_transmit = [i.cpu() for i in tensor_list_to_transmit] if CUDA else tensor_list_to_transmit
            Delta_C_i_to_transmit = [i.cpu() for i in Delta_C_i] if CUDA else Delta_C_i
            tensor_list_received = self.connection.root.aggregate(self.rank, self.sync_round_id, tensor_list_to_transmit, Delta_C_i_to_transmit)
            tensor_list_received = rpyc.classic.obtain(tensor_list_received)
            if CUDA:
                tensor_list_received = [tensor.cuda() for tensor in tensor_list_received]
            self.logging("receive global model")
            self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received)
            return True
        return False

    def try_get_global_model(self, dataset_size):
        tensor_list_received = self.connection.root.global_model(self.rank, dataset_size)
        tensor_list_received = rpyc.classic.obtain(tensor_list_received)
        if CUDA:
            tensor_list_received = [tensor.cuda() for tensor in tensor_list_received]
        self.transmit_interlayer.restore_model_from_tensor_list_received(tensor_list_received)

    def try_get_C_tensor_list(self):
        C_tensor_list_received = self.connection.root.C_tensor_list(self.rank)
        C_tensor_list_received = rpyc.classic.obtain(C_tensor_list_received)
        if CUDA:
            C_tensor_list_received = [tensor.cuda() for tensor in C_tensor_list_received]
        return C_tensor_list_received

    def try_start_train(self):
        return self.connection.root.start()
