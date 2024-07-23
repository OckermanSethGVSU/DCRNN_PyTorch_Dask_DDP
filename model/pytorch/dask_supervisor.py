import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from dcrnn_model import DCRNNModel
from loss import masked_mae_loss

from dask.distributed import LocalCluster
from dask.distributed import Client
from dask.array.lib.stride_tricks import sliding_window_view
from dask.distributed import wait as Wait
from dask.delayed import delayed
import dask.array as da
import dask.dataframe as dd
from dask_pytorch_ddp import dispatch, results

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, Sampler

import uuid
import pandas as pd
import math

class TrainDataset(Dataset):
    def __init__(self,x, y, lazy_batching=False):
         self.x = x 
         self.y = y 
        #  self.ycl = ycl
         self.lb = lazy_batching
    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.lb:
            x_sample = self.x[idx].compute()
            y_sample = self.y[idx].compute()
            # ycl_sample = self.ycl[idx].compute()
            
            # Convert to PyTorch tensors
            x_tensor = torch.from_numpy(x_sample)
            y_tensor = torch.from_numpy(y_sample)
            # ycl_tensor = torch.from_numpy(ycl_sample)

            return x_tensor, y_tensor
        
        return self.x[idx], self.y[idx]

class ValDataset(Dataset):
    def __init__(self, x,y, lazy_batching=False):
         self.x = x
         self.y = y
         self.lb = lazy_batching

    def __len__(self):
        # Return the number of samples
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.lb:
            x_sample = self.x[idx].compute()
            y_sample = self.y[idx].compute()
            
            # Convert to PyTorch tensors
            x_tensor = torch.from_numpy(x_sample)
            y_tensor = torch.from_numpy(y_sample)
           

            return x_tensor, y_tensor
        
        return self.x[idx], self.y[idx]
    

class DistributedBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_replicas = None,
        rank= None,
        shuffle = True,
        seed = 0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # Compute the number of batches
        dataset_size = len(dataset)
        self.num_batches = math.ceil(dataset_size / batch_size)
        
        if self.drop_last:
            self.num_batches = math.floor(dataset_size / batch_size)
            dataset_size = self.num_batches * batch_size  # Adjust size to be evenly divisible

        self.total_size = self.num_batches * batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # local_chunk_size = self.total_size // self.num_replicas
        # local_num_batches = self.num_batches // self.num_replicas
        
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            batch_indices = torch.randperm(self.num_batches, generator=g).tolist()
            # print("batch indicies: ", batch_indices)
        else:
            batch_indices = list(range(self.num_batches))

        # Convert batch indices to actual indices
        # indices = np.empty_like(batch_indices)
        # indicies = 
        # for batch_idx in batch_indices:
        #     start = batch_idx * self.batch_size
        #     end = min(start + self.batch_size, self.total_size)
        #     indices.extend(range(start, end))
        counter = 0
        indices = np.empty((self.total_size,), dtype=np.int32)
        
        for batch_idx in batch_indices:
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, self.total_size)
            indices[(counter * self.batch_size): min( (counter + 1 ) * self.batch_size, self.total_size)] = range(start, end)
            counter += 1
        
        # print("indcies: ", len(indices), len(indices) / self.batch_size)
        chunkSize = self.total_size // self.num_replicas
        # print("chunkSize: ", chunkSize)
        local_indx = indices[self.rank * (chunkSize):(self.rank + 1) * (chunkSize)]
        # Subsample for the current rank
        # indices = indices[self.rank * self.batch_size : self.total_size : self.num_replicas * self.batch_size]
        
        return iter(local_indx)

    def __len__(self) -> int:
        return len(self.__iter__())

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler to ensure shuffling is epoch-dependent."""
        self.epoch = epoch

class CustomBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.indices = data_source
        self.num_batches  = len(data_source)
        print("numBatch: ", self.num_batches)
        # print("len: ", len(data_source))
        # self.num_batches = len(data_source) // batch_size
        # self.indices = np.arange(self.num_batches)

    def __iter__(self):

        # print(self.data_source)
        # np.random.shuffle(self.indices)
        # batch_indices = np.empty((self.num_batches, self.batch_size), dtype=int)
        # for i, idx in enumerate(self.indices):
        #     start = idx * self.batch_size
        #     end = start + self.batch_size
        #     batch_indices[i] = np.arange(start, end)
        return iter( np.random.shuffle(self.indices))
        

    def __len__(self):
        return self.num_batches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def my_train(x_train=None, y_train=None, ycl_train=None, x_val=None, y_val=None, mean=None, std=None, 
             start_time=None, pre_end=None, 
             train_dict=None, model_dict=None, data_dict=None, 
             lazy_batching=False):
            worker_rank = int(dist.get_rank())
            # print("Top of my train: ", worker_rank, flush=True)
            device = f"cuda:{worker_rank % 4}"
            print(device)
            # print(train_dict)
            # device=None
            torch.cuda.set_device(worker_rank % 4)

            if not os.path.exists("logs/"):
                os.makedirs("logs/")
            if not os.path.exists("logs/info.log"):
                with open("logs/info.log", 'w') as file:
                    file.write('')
            log_dir = "logs/"
            writer = SummaryWriter('runs/' + log_dir)

            log_level =  "INFO"
            logger = utils.get_logger(log_dir, __name__, 'info.log', level=log_level)
            wait = 0

            lazy_batching = False
           

            from utils import load_graph_data
            sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(model_dict['graph_pkl_filename'])
            
            if not lazy_batching:
                x_train = x_train.compute()
                y_train = y_train.compute()
                # ycl_train = ycl_train.compute()

                x_val = x_val.compute()
                y_val = y_val.compute()
            
            
            train_dataset = TrainDataset(x_train, y_train, lazy_batching=lazy_batching)
            val_dataset = ValDataset(x_val, y_val, lazy_batching=lazy_batching)

            train_sampler = DistributedBatchSampler(train_dataset, data_dict['batch_size'], 
                                                    num_replicas=train_dict['npar'], rank=worker_rank,
                                                    drop_last=True)
            val_sampler = DistributedBatchSampler(val_dataset, data_dict['batch_size'], 
                                                    num_replicas=train_dict['npar'], rank=worker_rank,
                                                    drop_last=True, shuffle=False)
            # train_sampler = DistributedSampler(train_dataset, 
            #                                         num_replicas=train_dict['npar'], rank=worker_rank, shuffle=False)
            # val_sampler = DistributedSampler(val_dataset, 
            #                                         num_replicas=train_dict['npar'], rank=worker_rank,
            #                                         drop_last=True, shuffle=False)
            
            train_loader = DataLoader(train_dataset, batch_size=data_dict['batch_size'], sampler=train_sampler, shuffle=False, drop_last=True)
            train_per_epoch = (x_train.shape[0] // data_dict['batch_size']) // train_dict['npar']
            val_loader = DataLoader(val_dataset, batch_size=data_dict['batch_size'], sampler=val_sampler, shuffle=False, drop_last=True)
            val_per_epoch = (x_val.shape[0] // data_dict['batch_size']) // train_dict['npar']
            
            
            
            # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=False, drop_last=True)
            # train_per_epoch = len(train_loader)
            # val_sampler = DistributedSampler(val_dataset, num_replicas=args.npar, rank=worker_rank)
            # val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False, drop_last=True)
            # val_per_epoch = len(val_loader)

            scaler = utils.StandardScaler(mean=mean, std=std)

            if train_dict['load_path']:
                print("Loading saved state", flush=True)
                checkpoint = torch.load(train_dict['load_path'])
                model = DCRNNModel(adj_mx, logger, **model_dict)
                model = DDP(model, gradient_as_bucket_view=True).to(device)

                with torch.no_grad():
                    model = model.eval()



                   

                    # val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
                    

                    
                    # this is mimicking there setup graph function
                    for i, (x, y) in enumerate(val_loader):
                        
                        x = x.to(device).float()
                        y = y.to(device).float()
                        
                        x = x.permute(1, 0, 2, 3)
                        y = y.permute(1, 0, 2, 3)
                        batch_size = x.size(1)
                        x = x.view(model_dict['seq_len'], batch_size, model_dict['num_nodes'] * \
                                model_dict['input_dim'])
                        
                        y = y[..., :model_dict['output_dim']].view(model_dict['horizon'], batch_size,
                                                        model_dict['num_nodes'] * model_dict['output_dim'])

                        output = model(x)
                        
                        
                model.module.load_state_dict(checkpoint['model_state_dict'])

                start_epoch = checkpoint['epoch']
                start_epoch += 1
                batches_seen = checkpoint['batches_seen']
                optimizer = torch.optim.Adam(model.module.parameters(), lr=train_dict['base_lr'], eps=train_dict['epsilon'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_dict['steps'],
                                                                    gamma=train_dict['lr_decay_ratio'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                epochs = train_dict['epochs']
                # start_epoch, batches_seen, engine.optimizer = model.module.load_checkpoint(args.load_path, engine.optimizer)
            else:
                model = DCRNNModel(adj_mx, logger, **model_dict)
                model = DDP(model, gradient_as_bucket_view=True).to(device)

                # print(adj_mx)
                optimizer = torch.optim.Adam(model.module.parameters(), lr=train_dict['base_lr'], eps=train_dict['epsilon'])

                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_dict['steps'],
                                                                    gamma=train_dict['lr_decay_ratio'])
                # model = DDP(model, gradient_as_bucket_view=True).to(device)
                # model = DDP(model, gradient_as_bucket_view=True)
                

                logger.info('Start training ...')
                epochs = train_dict['epochs']
                start_epoch = 0
                batches_seen = 0
            
            if worker_rank == 0:
                print("Model created successfully; About to begin epochs", flush=True)

                if os.path.exists("per_epoch_stats.txt"):
                    pass
                else:
                    with open("per_epoch_stats.txt", "w") as file:

                            file.write(f"epoch,batches_seen,batches_seen,train_loss,val_loss,lr\n")
            
            

            
            # this will fail if model is loaded with a changed batch_size
            # num_batches = self._data['train_loader'].num_batch
            # self._logger.info("num_batches:{}".format(num_batches))

            # batches_seen = num_batches * self._epoch_num
            overall_t_loss = []
            overall_v_loss = []
            for epoch_num in range(start_epoch, epochs):
                
                model = model.train()
                train_sampler.set_epoch(epoch_num)
                val_sampler.set_epoch(epoch_num)
                
                # # shuffle the batches
                # train_iterator = self._data['train_loader'].get_iterator()
                # all_train = np.array([(x,y) for _, (x, y) in enumerate(train_iterator)])
                # permutation = np.random.permutation(all_train.shape[0])
                # all_train = all_train[permutation]
                
                
                losses = []
                t1 = time.time()

                for i, (x, y) in enumerate(train_loader):
                    
                    if worker_rank == 0:
                        print(f"\rEpoch {epoch_num} train batch {i + 1}/{train_per_epoch}", flush=True, end="")
                        print(batches_seen, flush=True)

                    
                    optimizer.zero_grad()

                    # x, y = DCRNNSupervisor._prepare_data(x, y)
                    x = x.to(device).float()
                    y = y.to(device).float()
                    logger.debug("X: {}".format(x.size()))
                    logger.debug("y: {}".format(y.size()))
                    x = x.permute(1, 0, 2, 3)
                    y = y.permute(1, 0, 2, 3)
                    batch_size = x.size(1)
                    x = x.view(model_dict['seq_len'], batch_size, model_dict['num_nodes'] * \
                               model_dict['input_dim'])
                    
                    y = y[..., :model_dict['output_dim']].view(model_dict['horizon'], batch_size,
                                                    model_dict['num_nodes'] * model_dict['output_dim'])
                    
                    output = model(x, y, batches_seen)

                    if batches_seen == 0:
                        # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                        optimizer = torch.optim.Adam(model.module.parameters(), lr=train_dict['base_lr'], eps=train_dict['epsilon'])
                    # loss = self._compute_loss(y, output)
                    
                    y_true = scaler.inverse_transform(y)
                    y_predicted = scaler.inverse_transform(output)
                    loss = masked_mae_loss(y_predicted, y_true)

                    
                    logger.debug(loss.item())

                    losses.append(loss.item())

                    batches_seen += 1
                    loss.backward()

                    # gradient clipping - this does it in place
                    torch.nn.utils.clip_grad_norm_(model.module.parameters(), train_dict['max_grad_norm'])
                    
                    optimizer.step()

                    if i == 3: break
                logger.info("epoch complete")
                lr_scheduler.step()
                logger.info("evaluating now!")

                # val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)
                with torch.no_grad():
                    model = model.eval()

                    # val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
                    val_losses = []

                    y_truths = []
                    y_preds = []

                    for i, (x, y) in enumerate(val_loader):
                        if worker_rank == 0:
                            print(f"\rEpoch {epoch_num} val batch {i + 1}/{val_per_epoch}", flush=True, end="")
                    
                        x = x.to(device).float()
                        y = y.to(device).float()
                        logger.debug("X: {}".format(x.size()))
                        logger.debug("y: {}".format(y.size()))
                        x = x.permute(1, 0, 2, 3)
                        y = y.permute(1, 0, 2, 3)
                        batch_size = x.size(1)
                        x = x.view(model_dict['seq_len'], batch_size, model_dict['num_nodes'] * \
                                model_dict['input_dim'])
                        
                        y = y[..., :model_dict['output_dim']].view(model_dict['horizon'], batch_size,
                                                        model_dict['num_nodes'] * model_dict['output_dim'])

                        output = model(x)
                        y_true = scaler.inverse_transform(y)
                        y_predicted = scaler.inverse_transform(output)
                        loss = masked_mae_loss(y_predicted, y_true)
                        val_losses.append(loss.item())

                        y_truths.append(y.cpu())
                        y_preds.append(output.cpu())
                        if i == 3: break

                    val_loss = np.mean(val_losses)
                    overall_v_loss.append(val_loss)
                    writer.add_scalar('{} loss'.format('val'), val_loss, batches_seen)

                    # y_preds = np.concatenate(y_preds, axis=1)
                    # y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

                    # y_truths_scaled = []
                    # y_preds_scaled = []
                    # # Talk to AMAL, this is never used
                    # for t in range(y_preds.shape[0]):
                    #     y_truth = scaler.inverse_transform(y_truths[t])
                    #     y_pred = scaler.inverse_transform(y_preds[t])
                    #     y_truths_scaled.append(y_truth)
                    #     y_preds_scaled.append(y_pred)

                # mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}
                t2 = time.time()

                writer.add_scalar('training loss',
                                        np.mean(losses),
                                        batches_seen)

                if worker_rank == 0:
                    overall_t_loss.append(np.mean(losses))
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                            '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                            np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                            (t2 - t1))
                    logger.info(message)
                    print(message)

                    
                    if not os.path.exists('models/'):
                        os.makedirs('models/')

                    checkpoint = {
                        'epoch': epoch_num,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
                        'batches_seen' : batches_seen
                    }


                    # print(self.state_dict())
                    torch.save(checkpoint, 'model_%d.pth' % epoch_num)
                    print(f"\ndaCheckpoint saved to {'model_%d.pth' % epoch_num}")
                    
                    with open("per_epoch_stats.txt", "a") as file:
                        # file.write(f"epoch, per_epoch_runtime, train_loss, val_loss, val_rmse, val_mape\n")
                        file.write(f"{epoch_num},{batches_seen},{t2 - t1},{np.mean(losses)},{val_loss},{lr_scheduler.get_last_lr()}\n")


                if False:
                    test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                    message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                            '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                            np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
                                            (end_time - start_time))
                    self._logger.info(message)
            end_time = time.time()
            if worker_rank == 0:

                with open("stats.txt", "a") as file:
                    file.write(f"training_time: {end_time - pre_end}\n")
                    file.write(f"total_time: {end_time - start_time}\n")

                    file.write(f"train_opt_loss: {min(overall_t_loss)}\n")
                    # file.write(f"train_opt_rmse: {min(overall_t_rmse)}\n")
                    # file.write(f"train_opt_mape: {min(overall_t_mape)}\n")

                    file.write(f"val_opt_loss: {min(overall_v_loss)}\n")
                    # file.write(f"val_opt_rmse: {min(his_rmse)}\n")
                    # file.write(f"val_opt_mape: {min(his_mape)}\n")
                # if val_loss < min_val_loss:
                #     wait = 0
                #     if save_model:
                #         model_file_name = self.save_model(epoch_num)
                #         self._logger.info(
                #             'Val loss decrease from {:.4f} to {:.4f}, '
                #             'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                #     min_val_loss = val_loss

                # elif val_loss >= min_val_loss:
                #     wait += 1
                #     if wait == patience:
                #         self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                #         break

class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        # self._log_dir = self._get_log_dir(kwargs)
        # self._writer = SummaryWriter('runs/' + self._log_dir)

        # log_level = self._kwargs.get('log_level', 'INFO')
        # self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        # self._data = utils.load_dataset(**self._data_kwargs)
        # self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        # dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        # model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
        # self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        
        # if self._epoch_num > 0:
        #     self.load_model()
        
    
    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch, optimizer = None, lr_scheduler = None, batches_seen = None):
        """
        Save the model checkpoint to a file.

        :param filepath: Path to the file where the checkpoint will be saved.
        :param epoch: Current epoch number.
        :param optimizer: Optimizer state to save along with model parameters (optional).
        """
        if not os.path.exists('models/'):
            os.makedirs('models/')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.dcrnn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
            'batches_seen' : batches_seen
        }


        # print(self.state_dict())
        torch.save(checkpoint, 'model_%d.pth' % epoch)
        print(f"\ndaCheckpoint saved to {'model_%d.pth' % epoch}")


        return 'models/epo%d.tar' % epoch

    # TODO fix
    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)

        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x)
                loss = self._compute_loss(y, output)
                losses.append(loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            return mean_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

  



    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        
        def readPD():
            df = pd.read_hdf(kwargs['h5'], key="df")
            df = df.astype('float32')
            df.index.freq='5min'  # Manually assign the index frequency
            df.index.freq = df.index.inferred_freq
            return df
        # steps is used in learning rate - will see if need to use it?
        def get_numeric_part(filename):
            return int(filename.split("_")[1].split(".")[0])
        
        
        if self._train_kwargs['load_path'] == "auto":
            current_dir = os.getcwd()
            files = os.listdir(current_dir)
            rel_files = [f for f in files if ".pth" in f]

            if len(rel_files) > 0:
                sorted_filenames = sorted(rel_files, key=get_numeric_part)
                self._train_kwargs['load_path'] = sorted_filenames[-1]
            else:
                self._train_kwargs['load_path'] = None



        start_time = time.time()
        if kwargs['mode'] == 'local':
            cluster = LocalCluster(n_workers=kwargs['npar'])
            client = Client(cluster)
        elif kwargs['mode'] == 'dist':
            client = Client(scheduler_file = f"cluster.info")
        else:
            print(f"{kwargs['mode']} is not a valid mode; Please enter mode as either 'local' or 'dist'")
            exit()
        
        
            
        
        dfs = delayed(readPD)()
        df = dd.from_delayed(dfs)
        min_val_loss = float('inf')
        

        num_samples, num_nodes = df.shape

        num_samples = num_samples.compute()
        
        x_offsets = np.sort(np.arange(-11, 1, 1))
        y_offsets = np.sort(np.arange(1, 13, 1))
        
        print("\rStep 1a Starting: df.to_dask_array", flush=True)
        data1 =  df.to_dask_array(lengths=True)
        # print(data1.shape)
        data1 = da.expand_dims(data1, axis=-1)
        data1 = data1.rechunk("auto")



        print("\rStep 1b Starting: Tiling", flush=True)
        print("kwargs at top: ", kwargs, flush=True)
        data2 = da.tile((df.index.values.compute() - df.index.values.compute().astype("datetime64[D]")) / np.timedelta64(1, "D"), [1, num_nodes, 1]).transpose((2, 1, 0))
        data2 = data2.rechunk((data1.chunks))
        
        
        # print("\rStep 1c Starting: Tiling", end="\n", flush=True)
        memmap_array = da.concatenate([data1, data2], axis=-1)
        

        del df
        # print("\rStep 1a Done; Step 1b Starting", flush=True)






        del data1 
        del data2

        
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
        total = max_t - min_t

        window_size = 12
        original_shape = memmap_array.shape

        
        # Define the window shape
        window_shape = (window_size,) + original_shape[1:]  # (12, 207, 2)

        # Use sliding_window_view to create the sliding windows
        sliding_windows = sliding_window_view(memmap_array, window_shape).squeeze()
        # time.sleep(15)
        # print(sliding_windows.compute().shape)
        # print(sliding_windows.compute())
        
        x_array = sliding_windows[:total]
        y_array = sliding_windows[window_size:]
        del memmap_array
        del sliding_windows





        num_samples = x_array.shape[0]
        num_test = round(num_samples * 0.2)
        num_train = round(num_samples * 0.7)
        num_val = num_samples - num_test - num_train

        

        x_train = x_array[:num_train]
        y_train = y_array[:num_train]
        # ycl_train = y_array[:num_train]

        # x_train = x_train
        # y_train = y_train
        # ycl_train = ycl_train

        # wait([x_train,y_train, ycl_train])


        # print("Step 3: Computing Mean and Std-Dev", flush=True)
        mean = x_train[..., 0].mean()
        std = x_train[..., 0].std()
        

        # print("Step 4a: Standardizing Train x Dataset",end="",  flush=True)
        x_train[..., 0] = (x_train[..., 0] - mean) / std
        y_train[..., 0] = (y_train[..., 0] - mean) / std
        # x_train = x_train)
        
        
        # print("\rStep 4b: Standardizing Train ycl Dataset",  flush=True)
        # ycl_train[..., 0] = (ycl_train[..., 0] - mean) / std
        # ycl_train = ycl_train)
        


        x_val = x_array[num_train: num_train + num_val]
        y_val = y_array[num_train: num_train + num_val]



        # print("Step 5: Standardizing Validation Dataset")
        x_val[..., 0] = (x_val[..., 0] - mean) / std
        y_val[..., 0] = (y_val[..., 0] - mean) / std
        
        # x_val = x_val)
        print("\rStep 1c: Concat, window, standardize" , flush=True)
        mean, std, x_train, y_train, x_val, y_val = client.persist([mean, std, x_train, y_train, x_val, y_val])
        
        
        Wait([mean, std, x_train, y_train, x_val, y_val])
        
        # time.sleep(30)
        mean = mean.compute()
        std = std.compute()


        pre_end = time.time()
        print(f"Preprocessing complete in {pre_end - start_time}; Training Starting")
        
        if os.path.exists("stats.txt"):
            with open("stats.txt", "a") as file:
                    file.write(f"pre_processing_time: {pre_end - start_time}\n")
        else:
            with open("stats.txt", "a") as file:
                file.write(f"pre_processing_time: {pre_end - start_time}\n")

        # x_train = x_train.compute()
        # y_train = y_train.compute()
        # ycl_train = ycl_train.compute()
        
        # x_val = x_val.compute()
        # y_val = y_val.compute()
        # wait([x_train, y_train, ycl_train, x_val, y_val])
        # time.sleep(60)
        del x_array
        del y_array
        

        
        
    

        # args = (x_train, y_train, ycl_train, x_val, y_val)
        if kwargs['mode'] == "dist":
            for f in ['utils.py', 'dcrnn_cell.py', 'dcrnn_model.py', 'loss.py', 'dask_supervisor.py']:
                client.upload_file(f)
        
        
        futures = dispatch.run(client, my_train, mean=mean, std=std, 
                               x_train=x_train, y_train=y_train, 
                               x_val=x_val, y_val=y_val, 
                               start_time=start_time, pre_end=pre_end, 
                               train_dict= self._train_kwargs, model_dict=self._model_kwargs, data_dict=self._data_kwargs,
                               lazy_batching = True,
                               backend="gloo")
        key = uuid.uuid4().hex
        rh = results.DaskResultsHandler(key)
        rh.process_results(".", futures, raise_errors=False)
        client.shutdown()
        
        
        
        

    def _prepare_data(x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
