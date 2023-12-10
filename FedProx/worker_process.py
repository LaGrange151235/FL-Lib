import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

import argparse, sys, os, copy, pprint, datetime, time, random, numpy, torch

sys.path.append(os.path.dirname(__file__)+"/..")

from model_manager import Model_Manager
from dataset_manager import Dataset_Manager
from sync_manager import Sync_Manager

CUDA = torch.cuda.is_available()
torch.set_printoptions(profile="full")

def model_2_tensor_list(model):
    tensor_list = []
    for param in model.parameters():
        tensor_list.append(param.data)
    return tensor_list

def model_2_flatten_tensor(model):
    tensor_list = []
    for param in model.parameters():
        tensor_list.append(param.data)
    flatten_tensor = [tensor.view(-1) for tensor in tensor_list]
    flatten_tensor = torch.cat(flatten_tensor, dim=0)
    return flatten_tensor

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

class Client:
    def __init__(self, local_profile, sync_profile):
        self.rank = local_profile['rank']
        
        ''' Model Initialization '''
        self.model_manager = Model_Manager(local_profile['model_name'])
        self.model = self.model_manager.load_model()
        self.optimizer = self.model_manager.get_optimizer()
        self.loss_func = self.model_manager.get_loss_func()
      
        ''' Dataset Initialization ''' 
        self.dataset_manager = Dataset_Manager(local_profile['dataset_profile'])
        self.training_dataloader = self.dataset_manager.get_training_dataloader()
        self.max_epoch = 1000000

        ''' Sync-related variables (the above part shall be able to run under local mode)'''
        self.sync_manager = Sync_Manager(self.model, self.rank, sync_profile)
        self.sync_frequency = sync_profile["sync_frequency"]

        self.dataset_size = self.dataset_manager.local_size

        # TODO: For FedProx
        self.prox_mu = 0.1
        self.init_model_params = None
                
        numpy.random.seed(self.rank)

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Client] '+str(string))
        sys.stdout.flush()
    
    def train(self):
        print('\n\n\t-------- START TRAINING --------\n')
        # TODO: fetch global model
        self.sync_manager.try_get_global_model(self.dataset_size)

        self.init_model_params = torch.cat([param.view(-1) for param in self.model.parameters()])

        self.logging("receive global model")
        while self.sync_manager.try_start_train() == False:
            time.sleep(0.01)

        iter_id, round_id, epoch_id  = 0, 0, 0
        while epoch_id < self.max_epoch:
            epoch_id += 1 
            self.logging('start epoch: %d' % epoch_id)
            for step, (b_x, b_y) in enumerate(self.training_dataloader):
                iter_id += 1
                if CUDA:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                self.optimizer.zero_grad()
                
                loss = self.loss_func(self.model(b_x), b_y)
                client_drift = torch.cat([param.view(-1) for param in self.model.parameters()]) - self.init_model_params
                client_drift = (self.prox_mu/2)*(torch.norm(client_drift))
                loss += client_drift
                
                loss.backward()
                self.optimizer.step()
                self.logging('finish local iteration: %d, loss: %.6f' % (iter_id, loss))
                
                if self.sync_manager.try_sync_model(iter_id):
                    round_id += 1                

            self.logging('finish epoch: %d\n' % epoch_id)

if __name__ == "__main__":
    ''' Parse arguments and create APF profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', type=str, default='/root')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=int, default=20000)
    parser.add_argument('--world_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--model_name', type=str, default='CNN_Cifar10')
    parser.add_argument('--dataset_name', type=str, default='Cifar10')
    parser.add_argument('--is_iid', type=int, default=0)
    parser.add_argument("--sync_frequency", type=int, default=100)
    parser.add_argument("--interlayer_type", type=str, default='Default')

    args = parser.parse_args()

    print('- Hypermeters: ')
    pprint.pprint(args)
    print()

    ''' A. Local Training Profile '''
    model_name = args.model_name
    dataset_name = args.dataset_name
    is_iid = args.is_iid
    local_profile = {
        'rank' : args.rank,
        'model_name' : model_name,
        'dataset_profile' : {
            'home' : args.home,
            'dataset_name' : dataset_name,
            'is_iid' : is_iid,
            'total_partition_number' : 1 if is_iid==0 else args.world_size,
            'partition_rank' : 0 if is_iid==0 else args.rank,
            'rank': args.rank,
            'world_size': args.world_size
        },
    }
    print('- Local Training Profile: ')
    pprint.pprint(local_profile)
    print()

    ''' B. Synchronization Profile '''
    sync_frequency = args.sync_frequency
    interlayer_type = args.interlayer_type
    sync_profile = {
        'sync_frequency' : sync_frequency,
        'interlayer_type' : interlayer_type,
        'server_ip': args.server_ip,
        'server_port': args.server_port,
    }
    print('- Sync Profile: ')
    pprint.pprint(sync_profile)
    print()
    
    ''' Launch Training '''
    client = Client(local_profile, sync_profile) # prepare local training environment
    client.train() 