import sys
import torch
import numpy as np

from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data_loader import LabelType

def accuracy(true, pred, **kwrgs):
    acc = sum(true == pred) / len(true)
    return acc


def IoU(true, pred, nclass = 3):
    """
        Intersection over union is a metric for semantic segmentation.
        It returns a IoU value for each class of our input tensors/arrays.
    """
    eps = sys.float_info.epsilon
    confusion_matrix = np.zeros((nclass, nclass))

    for i in range(len(true)):
        confusion_matrix[true[i]][pred[i]] += 1

    IoU = []
    for i in range(nclass):
        IoU.append((confusion_matrix[i, i] + eps) / (sum(confusion_matrix[:, i]) + sum(confusion_matrix[i, :]) - confusion_matrix[i, i] + eps))
    return np.array(IoU)

def metrics_selector(label_type, nclass):
    if label_type == LabelType.Classification:
        metric_fn = accuracy
        met_epoch = 0
    elif label_type == LabelType.Segmentation:
        metric_fn = IoU
        met_epoch = np.zeros(nclass)
    return metric_fn, met_epoch
    

def train_one_epoch(epoch_id, model, loader, device, optimizer, loss_fn, label_type, nclass = 4, model_uses_batch = True):
    '''
    Train the model with all the data once (an epoch)
    '''
    # Tell the model it's going to train
    model.train()

    loss_epoch = 0
    metric_fn, met_epoch = metrics_selector(label_type, nclass)

    # Iterate for the batches in the data loader
    for batch in loader:
        # Pass the batch to device (cuda)
        batch = batch.to(device)

        # Zero grad the optimizer
        optimizer.zero_grad()

        # Pass the data to the model
        if model_uses_batch:
            out = model.forward(batch.x.type(torch.float), batch.edge_index, batch.batch) 
        else:
            out = model.forward(batch.x.type(torch.float), batch.edge_index)

        # Now we pass the output and the labels to the loss function
        # We will use nll or cross entropy loss
        # These losses will need input (N, C) target (N); being C = num of classes, N = batch size
        
        # We read the label, transform into long tensor (needed by this loss function), pass to cuda device 
        segclass = batch.y
        label = segclass.type(torch.LongTensor).to(device) 

        # The reshape is needed to pass from a (N, 1) shape (automatically appears when doing
        # batch.y), to a (N) shape as we need; the output of the net is already (N, C) if it's properly built
        loss = loss_fn(out, torch.reshape(label, (-1,)))
        
        # Back propagation (compute gradients of the loss with respect to the weights in the model)
        loss.backward()
        # Gradient descent (update the optimizer)
        optimizer.step()

        loss_epoch += loss.item()


        pred = torch.reshape(out.argmax(dim=-1, keepdim=True), (-1,)).detach().cpu().numpy()
        true = torch.reshape(segclass, (-1,)).detach().cpu().numpy() 

        met_epoch += metric_fn(true, pred, nclass = nclass)
    
    loss_epoch = loss_epoch / len(loader)
    met_epoch  = met_epoch / len(loader)
    epoch_ = f"Train Epoch: {epoch_id}"
    loss_  = f"\t Loss: {loss_epoch:.6f}"
    print(epoch_ + loss_)

    return loss_epoch, met_epoch


def valid_one_epoch(model, loader, device, loss_fn, label_type, nclass = 4, model_uses_batch = True):
    # Set the model to evaluate
    model.eval()

    loss_epoch = 0
    metric_fn, met_epoch = metrics_selector(label_type, nclass)

    with torch.no_grad():
    # Iterate for the batches in the data loader
        for batch in loader:
            # Put batch into device (cuda)
            batch = batch.to(device)

            if model_uses_batch:
                out = model.forward(batch.x.type(torch.float), batch.edge_index, batch.batch)
            else:
                out = model.forward(batch.x.type(torch.float), batch.edge_index)

            segclass = batch.y
            label = segclass.type(torch.LongTensor).to(device) 

            # The reshape is needed to pass from a (N, 1) shape (automatically appears when doing
            # batch.y), to a (N) shape as we need; the output of the net is already (N, C) if it's properly built
            loss = loss_fn(out, torch.reshape(label, (-1,)))
            loss_epoch += loss.item()

            # For each node set the maximum argument to pick a class
            pred = torch.reshape(out.argmax(dim=-1, keepdim=True), (-1,)).detach().cpu().numpy()

            #Once again, the labels are shifted by 1 to match the prediction positions (explained in train fun)
            true = torch.reshape(segclass, (-1,)).detach().cpu().numpy() 
            
            met_epoch += metric_fn(true, pred, nclass = nclass)
            

        loss_epoch = loss_epoch / len(loader)
        met_epoch  = met_epoch / len(loader)
        loss_ = f"\t Validation Loss: {loss_epoch:.6f}"
        print(loss_)

    return loss_epoch, met_epoch

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def train_net(*,
              nepoch,
              train_data,
              valid_data,
              train_batch_size,
              valid_batch_size,
              net,
              device,
              optimizer,
              criterion,
              label_type,
              nclass,
              model_uses_batch,
              checkpoint_dir,
              tensorboard_dir,
              num_workers):
    """
        Trains the net nepoch times and saves the model anytime the validation loss decreases
    """
    loader_train = DataLoader(train_data,
                            batch_size = train_batch_size,
                            shuffle = True,
                            num_workers = num_workers,
                            drop_last = True,
                            pin_memory = False)
    loader_valid = DataLoader(valid_data,
                            batch_size = valid_batch_size,
                            shuffle = True,
                            num_workers = 1,
                            drop_last = True,
                            pin_memory = False)

    start_loss = np.inf
    writer = SummaryWriter(tensorboard_dir)
    for i in range(nepoch):
        train_loss, train_met = train_one_epoch(i, net, loader_train, device, optimizer, criterion, label_type, nclass = nclass, model_uses_batch = model_uses_batch)
        valid_loss, valid_met = valid_one_epoch(net, loader_valid, device, criterion, label_type, nclass = nclass, model_uses_batch = model_uses_batch)

        if valid_loss < start_loss:
            save_checkpoint({'state_dict': net.state_dict(),
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
        writer.flush()
    writer.close()

def predict_gen(test_data, net, batch_size, device, model_uses_batch, label_type):
    loader_test = DataLoader(test_data,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 1,
                            drop_last = False,
                            pin_memory = False)
    
    net.eval()
    softmax = torch.nn.Softmax(dim = 1)
    with torch.autograd.no_grad():
        for batch in loader_test:
            batch = batch.to(device)

            if model_uses_batch:
                out = net.forward(batch.x.type(torch.float), batch.edge_index, batch.batch) 
            else:
                out = net.forward(batch.x.type(torch.float), batch.edge_index)

            y_pred = softmax(out).cpu().detach().numpy()

            if label_type == LabelType.Classification:
                out_dict = dict(label = batch.y.cpu().detach().numpy(), 
                                dataset_id = batch.dataset_id.cpu().detach.numpu(), 
                                prediction = y_pred)
                
            elif label_type == LabelType.Segmentation:
                _, lengths = np.unique(batch.batch.cpu().detach().numpy(), return_counts=True)
                dataset_id = np.repeat(batch.dataset_id.cpu().detach().numpy(), lengths)

                out_dict = dict(dataset_id = dataset_id, 
                                coords = batch.coords.cpu().detach().numpy(), 
                                label  = batch.y.cpu().detach().numpy().flatten(), 
                                energy = batch.x.cpu().detach().numpy().flatten(), 
                                prediction = y_pred)
            yield out_dict