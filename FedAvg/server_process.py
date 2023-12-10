import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

import argparse, sys, os, copy, pprint, datetime, time, random, torch
import torch.nn.functional as F
from threading import Lock

sys.path.append(os.path.dirname(__file__)+"/..")

from model_manager import Model_Manager
from dataset_manager import Dataset_Manager

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

class AggregationService(Service):
    def __init__(self, args):
        self.client_number = args.client_number
        self.global_tensor_list = None
        self.client_ready = 0
        self.start_train = False
        self.dataset_size = [0]*self.client_number
        self.arrived = [0]*self.client_number
        self.tensor_buffer = []
        self.rank_buffer = []
        self.dataset_size_buffer = []
        
        # TODO: add global model
        self.model_manager = Model_Manager(args.model_name)
        self.model = self.model_manager.load_model()
        self.global_tensor_list = [None]
        tensor_list = []
        for param in self.model.parameters():
            tensor_list.append(param.data)
        self.global_tensor_list = copy.deepcopy(tensor_list)
        self.last_pushed_tensor_lists = copy.deepcopy(tensor_list)
        
        # TODO: add C-fraction selection
        self.C = args.C
        self.C_mod = args.C_mod
        self.round_list = [0]*self.client_number
        self.aggregation_round = 0
        if self.C_mod == "random":
            if self.C < 1:
                self.participants = random.sample(range(0, self.client_number-1), int(self.client_number*self.C))
            else:
                self.participants = range(self.client_number)
               
        # TODO: record last test time to calculate the test gap as round time
        self.last_test_time = 0

        # TODO: aggregation lock
        self.aggregation_lock = Lock()

        # TODO: connect to the test process
        self.test_mod = args.test_mod
        if self.test_mod == "independent":
            rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
            rpyc_config['allow_pickle'] = True
            rpyc_config['sync_request_timeout'] = 1500
            self.connection = rpyc.connect('localhost', args.test_process_port, config=rpyc_config)

        # TODO: load test_dataloader
        if self.test_mod == "inline":
            dataset_profile = {
                'dataset_name' : args.dataset_name,
                'is_iid': True,
                'total_partition_number' : 1,
                'partition_rank' : 0,
                'rank': 0,
                'world_size': 1,
            }
            self.dataset_manager = Dataset_Manager(dataset_profile)
            self.testing_dataloader = self.dataset_manager.get_testing_dataloader()
        
    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Server] '+str(string))
        sys.stdout.flush()

    def test(self, round_id):
        for i, param in enumerate(self.model.parameters()):
            if CUDA:
                param.data = self.global_tensor_list[i].cuda()
            else:
                param.data = self.global_tensor_list[i]
        positive_test_number = 0
        total_test_number = 0
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(self.testing_dataloader):
                if CUDA:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                test_result = self.model(test_x)
                loss_func = self.model_manager.get_loss_func()
                loss = loss_func(test_result, test_y)               
                pred_y = torch.max(test_result, 1)[1].data.cpu().numpy()
                positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                total_test_number += len(test_y)
        accuracy = positive_test_number / total_test_number
        self.logging(' - test - round_id: %d; accuracy: %.4f; loss: %.4f' % (round_id, accuracy, loss))
        
    def exposed_global_model(self, rank, dataset_size):
        self.dataset_size[rank] = dataset_size
        tensor_list = []
        for param in self.model.parameters():
            tensor_list.append(param.data)
        self.client_ready += 1
        if self.client_number == self.client_ready:
            self.start_train = True
        self.logging("send global model to rank %d." % (rank))
        return tensor_list
    
    def exposed_start(self):
        return self.start_train
    
    def exposed_aggregate(self, rank, round_id, tensor_list):
        tensor_list = rpyc.classic.obtain(tensor_list)
        self.aggregation_lock.acquire()
        self.round_list[rank] = round_id
        
        # TODO: C-fraction selection mechanism
        if self.C_mod == "fast":
            if self.round_list[rank] < self.aggregation_round:
                self.aggregation_lock.release()
                self.logging('return rank %d, round_id %d *(straggler)' % (rank, round_id))
                return self.global_tensor_list
        elif self.C_mod == "random":
            if rank not in self.participants:
                self.aggregation_lock.release()
                self.logging('return rank %d, round_id %d *(not participant)' % (rank, round_id))
                return tensor_list
        
        self.arrived[rank] = 1
        self.tensor_buffer.append(tensor_list)
        self.rank_buffer.append(rank)
        self.dataset_size_buffer.append(self.dataset_size[rank])
        if sum(self.arrived) == int(self.client_number * self.C):
            '''record aggregation info'''
            self.aggregation_round += 1
            seq = []
            for rank_ in range(self.client_number):
                if self.arrived[rank_] == 1:
                    seq.append(rank_)
            seq.sort()
            self.logging("aggregation_round: %d, participants: %s" % (self.aggregation_round, str(seq)))

            '''FedAvg aggregation'''
            aggregated_tensor_list = []
            for tensor_id, tensor_content in enumerate(self.tensor_buffer[0]):
                sum_tensor = torch.zeros(tensor_content.size())
                for i in range(len(self.tensor_buffer)):
                    sum_tensor+= (self.dataset_size_buffer[i]/sum(self.dataset_size_buffer))*self.tensor_buffer[i][tensor_id]
                aggregated_tensor = sum_tensor
                aggregated_tensor_list.append(aggregated_tensor)
            self.global_tensor_list = aggregated_tensor_list
            
            '''test'''
            test_time = time.time()
            test_gap = 0
            if self.last_test_time != 0:
                test_gap = test_time  - self.last_test_time
            self.last_test_time = test_time
            self.logging("aggregation_round: %d, test_gap: %.4f" % (self.aggregation_round, test_gap))
            if self.test_mod == "independent":
                self.connection.root.test(self.global_tensor_list, round_id)
            elif self.test_mod == "inline":
                self.test(round_id)

            '''reset'''
            self.tensor_buffer = []
            self.rank_buffer = []
            self.dataset_size_buffer = []
            self.arrived = [0]*self.client_number
            if self.C_mod == "random":
                if self.C < 1:
                    self.participants = random.sample(range(0, self.client_number-1), int(self.client_number*self.C))
                    self.logging("participants for next round: %s" % (str(self.participants)))
                else:
                    self.participants = range(self.client_number)
                    self.logging("participants for next round: %s" % (str(self.participants)))
                        
            self.aggregation_lock.release()
        else:
            self.aggregation_lock.release()
            while self.arrived[rank] == 1:
                time.sleep(0.01)
        
        self.logging('return rank %d, round_id %d' % (rank, round_id))
        return self.global_tensor_list

if __name__ == "__main__":
    ''' Parse arguments and create APF profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', type=str, default='/root')
    parser.add_argument('--port', type=int, default=20000)
    parser.add_argument('--client_number', type=int, default=5)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--test_mod', type=str, default="independent")
    parser.add_argument('--test_process_port', type=int, default=10000)
    parser.add_argument('--model_name', type=str, default='CNN_Cifar10')
    parser.add_argument('--dataset_name', type=str, default='Cifar10')
    parser.add_argument('--C', type=float, default=0.75)
    parser.add_argument('--C_mod', type=str, default="fast")

    args = parser.parse_args()
    print('- Hypermeters: ')
    pprint.pprint(args)
    print()

    service = AggregationService(args)
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    rpyc_config['sync_request_timeout'] = 1500
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()