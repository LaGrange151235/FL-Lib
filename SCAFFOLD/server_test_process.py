import rpyc
from rpyc import Service
from rpyc.utils.server import ThreadedServer

import argparse, sys, os, copy, datetime, time, torch
from threading import Lock, Thread

sys.path.append(os.path.dirname(__file__)+"/..")

from model_manager import Model_Manager
from dataset_manager import Dataset_Manager

torch.set_printoptions(precision=4)
CUDA = torch.cuda.is_available()

test_lock = Lock()
model = None
testing_dataloader = None

class TestService(Service):
    def __init__(self) -> None:
        self.logging("Test Process Start Successfully")

    def logging(self, string):
        print('['+str(datetime.datetime.now())+'] [Test Process] '+str(string))
        sys.stdout.flush()

    def exposed_test(self, tensor_list_to_restore, round_id):
        tensor_list_to_restore = rpyc.classic.obtain(tensor_list_to_restore)
        if CUDA:
            tensor_list_to_restore = [tensor.cuda() for tensor in tensor_list_to_restore]
        testing_thread = Thread(target=test_thread, args=(round_id, tensor_list_to_restore))
        testing_thread.start()

def test_thread(round_id, tensor_list_to_restore):
        global model, testing_dataloader, test_lock

        test_lock.acquire()
        accuracy = 0
        positive_test_number = 0.0
        total_test_number = 0.0
        ''' restore param '''
        for i, param in enumerate(model.parameters()):
            param.data = tensor_list_to_restore[i]

        t1 = time.time()
        with torch.no_grad():
            for step, (test_x, test_y) in enumerate(testing_dataloader):
                if CUDA:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                test_result = model(test_x)
                loss_func = model_manager.get_loss_func()
                loss = loss_func(test_result, test_y)               
                pred_y = torch.max(test_result, 1)[1].data.cpu().numpy()
                positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
                total_test_number += len(test_y)
        accuracy = positive_test_number / total_test_number
        t2 = time.time()

        logging('test_time: %.4f' % (t2-t1))
        logging(' - test - round_id: %d; accuracy: %.4f; loss: %.4f' % (round_id, accuracy, loss))
        test_lock.release()

def logging(string):
    print('['+str(datetime.datetime.now())+'] [Test Process] '+str(string))
    sys.stdout.flush()

if __name__ == "__main__":
    ''' Parse arguments and create profile. '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--trial_no', type=int, default=0)
    parser.add_argument('--remarks', type=str, default='Remarks Missing...')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--home', type=str)
    
    args = parser.parse_args()
    print('- Hypermeters: ')
    print(args)
    print()

    """ Local Training Profile """
    model_name, dataset_name = args.model_name, args.dataset_name
    dataset_profile = {
        'home' : args.home,
        'dataset_name' : dataset_name,
        'is_iid': True,
        'total_partition_number' : 1,
        'partition_rank' : 0,
        'rank': 0,
        'world_size': 1,
        'group_size': 1,
        'outlier_ratio': 0
    }

    ''' Model Initialization'''
    model_manager = Model_Manager(model_name)
    model = model_manager.load_model()

    ''' Dataset Initialization '''
    dataset_manager = Dataset_Manager(dataset_profile)
    testing_dataloader = dataset_manager.get_testing_dataloader()

    service = TestService()
    rpyc_config = rpyc.core.protocol.DEFAULT_CONFIG
    rpyc_config['allow_pickle'] = True
    rpyc_config['sync_request_timeout'] = 1500
    server = ThreadedServer(service, port=args.port, protocol_config=rpyc_config)
    server.start()