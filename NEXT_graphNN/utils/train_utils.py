import sys
import torch
import inspect
import numpy as np

from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


from .data_loader import LabelType

class Metrics():
    def __init__(self, labeltype, nclass = 3):
        super(Metrics, self).__init__()
        '''
        Here there are the different definitions of metrics, and which label, function and value container are used.
        '''

        def accuracy(true, pred):
            acc = sum(true == pred) / len(true)
            return acc

        def IoU(true, pred, nclass = nclass):
            """
            Intersection over union is a metric for semantic segmentation.
            It returns a IoU value for each class of our input tensors/arrays.
            It is inside the init of the class so that we have already the desired
            nclass from the moment we create the class
            """
            eps = sys.float_info.epsilon
            confusion_matrix = np.zeros((nclass, nclass))

            for i in range(len(true)):
                confusion_matrix[true[i]][pred[i]] += 1

            iou = []
            for i in range(nclass):
                iou.append((confusion_matrix[i, i] + eps) / (sum(confusion_matrix[:, i]) + sum(confusion_matrix[i, :]) - confusion_matrix[i, i] + eps))
            return np.array(iou)
        
        if labeltype == LabelType.Classification:
            self.label_name = 'binclass'
            self.met_epoch  = 0
            self.metric_fn  = accuracy

        elif labeltype == LabelType.Segmentation:
            self.label_name = 'y'
            self.met_epoch  = np.zeros(nclass)            
            self.metric_fn  = IoU
    

def train_one_epoch(epoch_id, 
                    model, 
                    loader, 
                    device, 
                    optimizer, 
                    loss_fn, 
                    label_type = LabelType.Classification, 
                    nclass = 2):
    # Tell the model it's going to train
    model.train()
    # Initialize the loss value container 
    loss_epoch = 0
    # Pick the desired metrics (kind of label, value container and function)
    metrics = Metrics(label_type, nclass = nclass)
    label_name, met_epoch, metric_fn = metrics.label_name, metrics.met_epoch, metrics.metric_fn

    # Iterate for the batches in the data loader
    for batch in loader:
        # Pass the batch to desired device (cpu/cuda)
        batch = batch.to(device)
        # 1. Zero grad the optimizer
        optimizer.zero_grad()
        # 2. Pass the data to the model
        out = model.forward(batch) 
        # Pick the desired target
        label = torch.reshape(torch.tensor(batch[label_name], dtype = torch.long), (-1,)).to(device)
        # 3. Compute the loss comparing the output of the model with the target
        loss = loss_fn(out, label)
        # 4. Back propagation (compute gradients of the loss with respect to the weights in the model)
        loss.backward()
        # 5. Gradient descent (update the optimizer) (4)
        optimizer.step()

        # sum loss to get at the end the average loss per epoch
        loss_epoch += loss.item()

        # Compute the metrics with the metrics function 
        true = label.detach().cpu().numpy() 
        pred = out.argmax(dim=-1).detach().cpu().numpy()
        met_epoch += metric_fn(true, pred)

    # Average the loss and metrics for the whole epoch
    loss_epoch = loss_epoch / len(loader)
    met_epoch  = met_epoch  / len(loader)

    # Print the values
    epoch_ = f"Train Epoch: {epoch_id}"
    loss_  = f"\t Loss: {loss_epoch:.6f}"
    met_   = f"\t Metr: {met_epoch:.6f}"
    print(epoch_ + loss_ + met_)
    
    return loss_epoch, met_epoch


def valid_one_epoch(model, 
                    loader, 
                    device, 
                    loss_fn, 
                    label_type = LabelType.Classification, 
                    nclass = 2):
    # Set the model to evaluate
    model.eval()

    loss_epoch = 0
    metrics = Metrics(label_type, nclass = nclass)
    label_name, met_epoch, metric_fn = metrics.label_name, metrics.met_epoch, metrics.metric_fn

    # Freeze the gradients
    with torch.no_grad():
    # Iterate for the batches in the data loader
        for batch in loader:
            # Put batch into device (cpu/cuda)
            batch = batch.to(device)
            # Pass the data through the model
            out = model.forward(batch)
            # Get the target
            label = torch.reshape(torch.tensor(batch[label_name], dtype = torch.long), (-1,)).to(device)
            # Compute the loss
            loss = loss_fn(out, label)
            # sum loss to get at the end the average loss per epoch
            loss_epoch += loss.item()

            # compute the metrics
            true = label.detach().cpu().numpy() 
            pred = out.argmax(dim=-1).detach().cpu().numpy()
            met_epoch += metric_fn(true, pred)
            
        loss_epoch = loss_epoch / len(loader)
        met_epoch  = met_epoch / len(loader)
        loss_  = f"\t Validation Loss: {loss_epoch:.6f}"
        met_   = f"\t Metr: {met_epoch:.6f}"
        print(loss_, met_)

    return loss_epoch, met_epoch

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def get_name_of_scheduler(scheduler):
    """
    Get the name of the scheduler.
    """
    for name, obj in lr_scheduler.__dict__.items():
        if inspect.isclass(obj):
            if isinstance(scheduler, obj):
                return name
    return None

def train_net(*,
              nepoch,
              train_dataset,
              valid_dataset,
              train_batch_size,
              valid_batch_size,
              num_workers,
              model,
              device,
              optimizer,
              criterion,
              scheduler,
              checkpoint_dir,
              tensorboard_dir,
              label_type = LabelType.Classification,
              nclass = 2):
    """
        Trains the net nepoch times and saves the model anytime the validation loss decreases
    """
    loader_train = DataLoader(train_dataset,
                            batch_size = train_batch_size,
                            shuffle = True,
                            num_workers = num_workers,
                            drop_last = True,
                            pin_memory = False)
    loader_valid = DataLoader(valid_dataset,
                            batch_size = valid_batch_size,
                            shuffle = True,
                            num_workers = 1,
                            drop_last = True,
                            pin_memory = False)

    start_loss = np.inf
    writer = SummaryWriter(tensorboard_dir)
    for i in range(nepoch):
        train_loss, train_met = train_one_epoch(i, model, loader_train, device, optimizer, criterion, label_type = label_type, nclass = nclass)
        valid_loss, valid_met = valid_one_epoch(   model, loader_valid, device,            criterion, label_type = label_type, nclass = nclass)
        if scheduler:
            if get_name_of_scheduler(scheduler) == 'ReduceLROnPlateau':
                scheduler.step(valid_loss)
            else:
                scheduler.step()

        if valid_loss < start_loss:
            save_checkpoint({'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}, f'{checkpoint_dir}/net_checkpoint_{i}.pth.tar')
            start_loss = valid_loss

        writer.add_scalar('loss/train', train_loss, i)
        writer.add_scalar('loss/valid', valid_loss, i)
        if label_type == LabelType.Segmentation:
            for k, iou in enumerate(train_met):
                writer.add_scalar(f'iou/train_{k}class', iou, i)
            for k, iou in enumerate(valid_met):
                writer.add_scalar(f'iou/valid_{k}class', iou, i)
        elif label_type == LabelType.Classification:
            writer.add_scalar('acc/train', train_met, i)
            writer.add_scalar('acc/valid', valid_met, i)
        learning_rate = optimizer.param_groups[0]['lr']
        writer.add_scalar('lr', learning_rate, i)
        print('lr = ', learning_rate)
        writer.flush()
    writer.close()

def predict_gen(test_data, model, batch_size, device, label_type = LabelType.Classification, nclass = 2):
    loader_test = DataLoader(test_data,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 1,
                            drop_last = False,
                            pin_memory = False)
    
    model.eval()
    softmax = torch.nn.Softmax(dim = 1)
    label_name = Metrics(label_type, nclass = nclass).label_name
    with torch.autograd.no_grad():
        for batch in loader_test:
            batch = batch.to(device)

            out = model.forward(batch)

            y_pred = softmax(out).cpu().detach().numpy()

            if label_type == LabelType.Classification:
                mean_xbin, mean_ybin, mean_zbin = torch.stack([data.coords.float().mean(axis = 0) for data in batch.to_data_list()]).t().tolist()
                maxE_xbin, maxE_ybin, maxE_zbin = torch.stack([data.coords[data.x[:, 0].argmax()] for data in batch.to_data_list()]).t().tolist()
                out_dict = dict(file_id    = batch.fnum.detach().cpu().numpy(), 
                                dataset_id = batch.dataset_id.detach().cpu().numpy(), 
                                binclass   = batch[label_name], #already a list 
                                num_nodes  = batch.batch.bincount().detach().cpu().numpy(), #add number of nodes
                                mean_xbin  = mean_xbin,
                                mean_ybin  = mean_ybin,
                                mean_zbin  = mean_zbin,
                                maxE_xbin  = maxE_xbin, 
                                maxE_ybin  = maxE_ybin,
                                maxE_zbin  = maxE_zbin,
                                prediction = y_pred)
                
            elif label_type == LabelType.Segmentation:
                _, lengths = np.unique(batch.batch.detach().cpu().numpy(), return_counts=True)
                dataset_id = np.repeat(batch.dataset_id.detach().cpu().numpy(), lengths)
                file_id    = np.repeat(batch.fnum.detach().cpu().numpy(), lengths)
                binclass   = np.repeat(batch.binclass, lengths)

                out_dict = dict(file_id    = file_id,
                                dataset_id = dataset_id, 
                                binclass   = binclass,
                                coords     = batch.coords.detach().cpu().numpy(), 
                                energy     = batch.x[:,0].detach().cpu().numpy().flatten(), 
                                label      = batch[label_name].detach().cpu().numpy().flatten(), 
                                prediction = y_pred)
            yield out_dict