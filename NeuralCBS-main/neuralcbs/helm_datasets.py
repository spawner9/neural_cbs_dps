import os
import torch
import numpy as np
from bisect import bisect
from torch.utils.data import Dataset, DataLoader  

class Speed_Norm:
    def __init__(self, device='cpu'):
        self.mean = torch.Tensor([1492.3629]).to(device)
        self.std = torch.Tensor([22.6009]).to(device)
    def encoder(self, input):
        return (input - self.mean)/self.std
    def decoder(self, input):
        return input*self.std+self.mean
    
class U_homo_Norm:
    def __init__(self, device='cpu'):
        self.mean = torch.Tensor([ 0.0623, -0.0020]).view(1,-1,1,1).to(device) 
        self.std = torch.Tensor([323.6842, 324.1692]).view(1,-1,1,1).to(device)
    def encoder(self, input):
        return (input - self.mean)/self.std
    def decoder(self, input):
        return input*self.std+self.mean

class Field_Norm:
    def __init__(self):
        self.mean = np.array([0.0837, 0.0080])[:,None,None]
        self.std = np.array([324.0742, 324.5560])[:,None,None]
    def encoder(self, input):
        return (input - self.mean)/self.std
    def decoder(self, input):
        return input*self.std+self.mean
    
class Speed_Norm:
    def __init__(self):
        self.mean = np.array([1492.3629])
        self.std = np.array([22.6009])
    def encoder(self, input):
        return (input - self.mean)/self.std
    def decoder(self, input):
        return input*self.std+self.mean
    
class U_homo_Norm:
    def __init__(self):
        self.mean = np.array([ 0.0623, -0.0020])[None,None,:]
        self.std = np.array([323.6842, 324.1692])[None,None,:]
    def encoder(self, input):
        return (input - self.mean)/self.std
    def decoder(self, input):
        return input*self.std+self.mean
    

class File_Loader(Dataset):
    def __init__(self, data_paths, target_paths,u_homo_path):
        self.data_paths = data_paths
        self.target_paths = target_paths
        self.start_indices_data = [0] * len(data_paths)
        self.start_indices_target = [0] * len(target_paths)
        self.data_count = 0
        self.data_count_target = 0 

        for index in range(len(data_paths)):
            self.start_indices_data[index] = self.data_count
            self.data_count+=1
        for index in range(len(target_paths)):
            self.start_indices_target[index] = self.data_count_target
            self.data_count_target+=8
        
        self.src = np.load(u_homo_path)
        print('data_count',self.data_count,'data_count_target',self.data_count_target)
        self.field_norm = Field_Norm()
        self.sos_norm = Speed_Norm()
        self.src_norm = U_homo_Norm()
    
    def __len__(self):
        return self.data_count_target
    
    def __getitem__(self, index) :
        memmap_index = bisect(self.start_indices_target, index) - 1 
        index_in_memmap = index - self.start_indices_target[memmap_index] 
        index_data = index//32
        index_src = index - index_data*32
        data = np.load(self.data_paths[index_data], mmap_mode='r')
        data = data[np.newaxis,:,:]
        target = np.load(self.target_paths[memmap_index],mmap_mode='r')[index_in_memmap]
        target =  np.concatenate((np.real(target)[np.newaxis,:,:],np.imag(target)[np.newaxis,:,:]),axis = 0)
        target = self.field_norm.encoder(target)
        src = self.src[index_src]
        src = self.src_norm.encoder(src)
        data = self.sos_norm.encoder(data)
        # theta = (index_src/32*2*np.pi)*np.ones((src.shape[0],src.shape[1],1))
        # theta = np.concatenate((theta, src), axis=-1)# 480*480*3

        return (torch.tensor(data,dtype=torch.float), # 1*480*480
                torch.tensor(src,dtype = torch.float).permute(2,0,1).contiguous(), # 2*480*480
                torch.tensor(target, dtype=torch.float))# 2*480*480
class GettingLists(object):
    def __init__(self,train_num = 6300,
                    valid_num = 900 ,
                    PATH_data = 'lbs', 
                    PATH_target = 'lbs',
                    PATH_src = 'lbs',
                    workers=20,
                    batchsize= int(2000),
                    num_folder=4,
                    ):
        super(GettingLists, self).__init__()
        self.PATH_data = PATH_data
        self.PATH_target = PATH_target
        self.PATH_src = PATH_src
        self.batchsize = batchsize
        self.workers = workers
        self.train_num = train_num
        self.valid_num = valid_num
        self.total_num = train_num+valid_num
        # self.velo_list = np.array([os.path.join(self.PATH_data,f'dataset_train_{i+1}/speed',f'train_{k}.npy') for i in range(0,8) for k in range(1+i*900,1+(i+1)*900)])
        # self.target_list = np.array([os.path.join(self.PATH_data,f'dataset_train_{i+1}/field',f'train_{k}_{j}.npy') for i in range(0,8) for k in range(1+i*900,1+(i+1)*900) for j in range(1,5)])
        self.velo_list = []
        self.target_list = []
        # get speed list
        for i in range(num_folder):
            speed_folder_dir = os.path.join(self.PATH_data, f'dataset_train_{i+1}/speed')
            speed_files_name = os.listdir(speed_folder_dir)
            for speed_file in speed_files_name:
                self.velo_list.append(os.path.join(speed_folder_dir, speed_file))
        # get field list
        for i in range(num_folder):
            field_folder_dir = os.path.join(self.PATH_data, f'dataset_train_{i+1}/field')
            field_index = [int(path.split("/")[-1].split("_")[-1].split(".")[0]) for path in self.velo_list if f"dataset_train_{i+1}" in path]
            for index in field_index:
                for j in range(4):
                    self.target_list.append(os.path.join(field_folder_dir, f'train_{index}_{j+1}.npy'))
        

    def get_list(self, do):
        if do == 'train':
            # in_limit_train= np.array([os.path.join(self.PATH_data,f'train_{k}.npy') for k in  self.velo_list_train ])
            # out_limit_train = np.array([os.path.join(self.PATH_target,f'train_{k}_{i}.npy') for k in  self.pressure_list_train for i in self.num_list])
            in_limit_train = np.array(self.velo_list[:self.train_num])
            out_limit_train= np.array(self.target_list[:self.train_num*4])
            
            return in_limit_train, out_limit_train
        elif do == 'validation':
            # in_limit_valid= np.array([os.path.join(self.PATH_data,f'train_{k}.npy') for k in  self.velo_list_test ])
            # out_limit_valid = np.array([os.path.join(self.PATH_target,f'train_{k}_{i}.npy') for k in  self.pressure_list_test for i in self.num_list]) 
            in_limit_valid = np.array(self.velo_list[self.train_num:self.train_num+self.valid_num])
            out_limit_valid= np.array(self.target_list[self.train_num*4:self.train_num*4+self.valid_num*4])
            return  in_limit_valid, out_limit_valid
        elif do =='test':
            # in_limit_test= np.array([os.path.join(self.PATH_data,f'train_{k}.npy') for k in  self.velo_list_test ])
            # out_limit_test = np.array([os.path.join(self.PATH_target,f'train_{k}_{i}.npy') for k in  self.pressure_list_test for i in self.num_list]) 
            in_limit_test = np.array(self.velo_list[self.train_num:self.train_num+self.valid_num])
            out_limit_test= np.array(self.target_list[self.train_num*4:self.train_num*4+self.valid_num*4])
            return in_limit_test, out_limit_test  
        
    def __call__(self, do = 'train'):
        return self.get_list(do)
    def get_dataloader(self,do):
        workers = self.workers
        batchsize = self.batchsize
        if do == 'train':
            list_x_train, list_y_train = self.__call__('train')
            list_x_valid, list_y_valid = self.__call__('validation')
            Train_Data_set = File_Loader(list_x_train,list_y_train, u_homo_path = self.PATH_src)
            Valid_Data_set = File_Loader(list_x_valid,list_y_valid, u_homo_path = self.PATH_src)
            train_loader = DataLoader(dataset = Train_Data_set, 
                                    shuffle = True, 
                                    batch_size = batchsize,
                                    num_workers= workers)
            valid_loader = DataLoader(dataset = Valid_Data_set, 
                                    shuffle = False, 
                                    batch_size =batchsize,
                                    num_workers= workers)
            # torch.save(normal,'normal.pt')
            return train_loader, valid_loader
        elif do == 'test':
            list_x_test, list_y_test = self.__call__('test')
            Test_Data_set = File_Loader(list_x_test, list_y_test, u_homo_path = self.PATH_src)
            test_loader = DataLoader(dataset = Test_Data_set, 
                                    shuffle = False, 
                                    batch_size = batchsize,
                                    num_workers= workers)
            return test_loader

def datasetFactory(do='train',
                   train_num=3000,
                   valid_num=500,
                   batchsize=16,
                   workers=20,
                   PATH_data='/work/helm_data',
                   PATH_target='/work/helm_data',
                   PATH_src='lbs',
                   num_folder=4):
    gl = GettingLists(train_num = train_num,
                     valid_num = valid_num,
                     PATH_data = PATH_data,
                     PATH_target= PATH_target,
                     PATH_src=PATH_src,
                     workers=workers,
                     batchsize = batchsize,
                     num_folder=num_folder)
    return gl.get_dataloader(do = do)


if __name__ == "__main__":
    import torch


    train_loader, val_loader = datasetFactory(do='train',
                                              train_num=3000,
                                              valid_num=500,
                                              batchsize=16,
                                              PATH_data='/work/helm_data',
                                              PATH_target='/work/helm_data',
                                              PATH_src='/work/helm_data/u_homo/u_homo.npy',
                                              workers=20,
                                              num_folder=4)
    data = next(iter(train_loader))
    for d in data:
        print(d.max())