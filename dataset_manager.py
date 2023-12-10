from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import sys, pprint, datetime, random, collections, numpy, torch

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label

class KWSconstructor(Dataset):
    def __init__(self, root, transform=None):
        f = open(root, 'r')
        data = []
        self.targets = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append((info[0], int(info[1])))
            self.targets.append(int(info[1]))
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        f, label = self.data[index]
        feature = numpy.loadtxt(f)
        feature = numpy.reshape(feature, (50, 10))
        feature = feature.astype(numpy.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label
 
    def __len__(self):
        return len(self.data)

class Dataset_Manager: 
    def __init__(self, dataset_profile): # dataset_name, is_iid, total_partition_number, rank):
        self.dataset_name = dataset_profile['dataset_name']
        self.is_iid = dataset_profile['is_iid']
        self.total_partition_number = dataset_profile['total_partition_number'] # world_size
        self.partition_rank = dataset_profile['partition_rank'] # rank
        self.rank = dataset_profile['rank'] # rank
        self.world_size = dataset_profile['world_size'] # world_size
        
        self.batch_size = 100 if dataset_profile['dataset_name'] != 'ImageNet' else 32
        self.training_dataset = self.get_training_dataset()
        self.testing_dataset = self.get_testing_dataset()

        self.class_num = len(list(set(numpy.array(self.training_dataset.targets))))
        
        self.dirichlet_alpha = 0.5
        self.local_size = len(self.training_dataset) / self.world_size

        self.logging('create dataset') # no special hyperparameter here for different dataset types

    def logging(self, string, hyperparameters=None):
        print('['+str(datetime.datetime.now())+'] [Dataset Manager] '+str(string))
        if hyperparameters != None:
            pprint.pprint(hyperparameters)
        sys.stdout.flush()

    def get_training_dataset(self):
        if self.dataset_name == 'Mnist':
            dataset = datasets.MNIST(root='./Datasets/mnist/', train=True, transform=transforms.ToTensor(), download=True)
        if self.dataset_name == 'EMnist':
            dataset = datasets.EMNIST(root='./Datasets/EMnist', split='balanced',train=True, download=True, transform=transforms.ToTensor())
        if self.dataset_name == 'Cifar10':
            dataset = datasets.CIFAR10(root='./Datasets/cifar10/', train=True, transform=transforms.ToTensor(), download=True)
        if self.dataset_name == 'KWS':
            dataset = KWSconstructor(root='./Datasets/kws/index_train.txt', transform=None)
        if self.dataset_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            dataset = datasets.ImageFolder('/data/imagenet/train', transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,]))
        return dataset

    def get_testing_dataset(self):
        if self.dataset_name == 'Mnist':
            dataset = datasets.MNIST(root='./Datasets/mnist/', train=False, transform=transforms.ToTensor())
        if self.dataset_name == 'EMnist':
            dataset = datasets.EMNIST(root='./Datasets/EMnist', split='balanced',train=False, download=True, transform=transforms.ToTensor())
        if self.dataset_name == 'Cifar10':
            dataset = datasets.CIFAR10(root='./Datasets/cifar10/', train=False, transform=transforms.ToTensor())
        if self.dataset_name == 'KWS':
            dataset = KWSconstructor(root='./Datasets/kws/index_test.txt', transform=None)
        if self.dataset_name == 'ImageNet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            dataset = datasets.ImageFolder('/data/imagenet/validate', transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,]))
        return dataset

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        numpy.random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def get_index_partition_1(self):
        num_shards, num_imgs = 200, len(self.training_dataset)//200
        idx_shard = [i for i in range(num_shards)]
        local_idxs = []
        labels = numpy.array(self.training_dataset.targets)
        sorted_idxs = numpy.argsort(labels)
                
        rand_set = set(numpy.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        self.logging("choose shards: %s, shards number: %d, shards size: %d" % (str(rand_set), num_shards, num_imgs))
        for rand in rand_set:
            local_idxs = numpy.concatenate((local_idxs, sorted_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        return local_idxs
    
    def get_dirichlet(self):
        composition_ratio = numpy.random.dirichlet(tuple(self.dirichlet_alpha for _ in range(self.class_num)), self.world_size)
        return (composition_ratio*self.local_size).astype(int)

    def get_index_partition_2(self):
        local_idxs = numpy.array([]).astype(int)
        self.set_seed(0)
        labels = numpy.array(self.training_dataset.targets)
        sorted_idxs = numpy.argsort(labels)
        composition = self.get_dirichlet()
        class_accumulate = numpy.sum(composition[:self.rank], axis=0)
        composition = composition[self.rank]
        self.logging(composition)
        self.logging('local dataset composition: %s' % (str(composition)))
        class_pool_size = int(len(self.training_dataset) / self.class_num)
        for i in range(self.class_num):
            client_class_index = numpy.array([]).astype(int)
            class_accumulate[i] = class_accumulate[i]%class_pool_size
            while class_accumulate[i] + composition[i] > class_pool_size:
                client_class_index = numpy.concatenate((client_class_index, sorted_idxs[class_pool_size*i+class_accumulate[i]:class_pool_size*(i+1)]),axis=0)
                composition[i] -= (class_pool_size-class_accumulate[i])
                class_accumulate[i] = 0
            client_class_index = numpy.concatenate((client_class_index, sorted_idxs[class_pool_size*i+class_accumulate[i]:class_pool_size*i+class_accumulate[i]+composition[i]]),axis=0)
            local_idxs = numpy.concatenate((local_idxs, client_class_index), axis = 0)
            local_idxs = local_idxs.astype(int)
        return local_idxs

    def get_index_partition_3(self):
        local_size = len(self.training_dataset)//self.world_size
        labels = numpy.array(self.training_dataset.targets)
        sorted_idxs = numpy.argsort(labels)
        local_idxs = sorted_idxs[self.rank*local_size:(self.rank+1)*local_size]
        return local_idxs   
        
    def get_training_dataloader(self):
        if self.is_iid == 0:
            self.set_seed(self.partition_rank)
            training_dataloader = DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
            return training_dataloader
        if self.is_iid == 1:
            self.set_seed(self.rank)
            index_partition = self.get_index_partition_1()
            training_dataloader = DataLoader(DatasetSplit(self.training_dataset, index_partition), batch_size=self.batch_size, shuffle=True)
            return training_dataloader
        if self.is_iid == 2:
            self.set_seed(0)
            index_partition = self.get_index_partition_2()
            training_dataloader = DataLoader(DatasetSplit(self.training_dataset, index_partition), batch_size=self.batch_size, shuffle=True)
            return training_dataloader
        if self.is_iid == 3:
            self.set_seed(0)
            index_partition = self.get_index_partition_3()
            training_dataloader = DataLoader(DatasetSplit(self.training_dataset, index_partition), batch_size=self.batch_size, shuffle=True)
            return training_dataloader

    def get_testing_dataloader(self):
        testing_dataloader = DataLoader(dataset=self.testing_dataset, batch_size=self.batch_size, shuffle=True)
        return testing_dataloader
