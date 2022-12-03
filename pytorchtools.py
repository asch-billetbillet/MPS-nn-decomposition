import numpy as np
import torch
from carbontracker.tracker import CarbonTracker


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def test_model(model, dataloader, resnet=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tracker = CarbonTracker(epochs=1)
    model.to(device)

    # Validate model
    with torch.no_grad():
        tracker.epoch_start()
        model.eval()
        n_correct = 0
        n_samples = 0
        
        for images, labels in dataloader:
            # Flatten MNIST images to 784 long vector
            if resnet:
                images = images.to(device)
            else:
                images = images.reshape(images.shape[0], -1).to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,predictions = torch.max(outputs, 1)

            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

        tracker.epoch_end()
        accuracy = n_correct/n_samples
        print(f'{accuracy =:.3f}')
        torch.cuda.empty_cache()
        return accuracy, tracker

def train_model(model, trainloader, validloader, optimizer, criterion, n_epochs, patience, save_path, resnet=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)

    for epoch in range(n_epochs):
            
        # Train the model
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            if resnet:
                inputs = inputs.to(device)
            else:
                inputs = torch.flatten(inputs, start_dim=1).to(device)
            target = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validate the model
        model.eval()
        with torch.no_grad():
            for data in validloader:
                inputs, labels = data
                if resnet:
                    inputs = inputs.to(device)
                else:
                    inputs = torch.flatten(inputs, start_dim=1).to(device)
                target = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, target)
                valid_losses.append(loss.item())
        
        # print training/validation statistics 
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch+1:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                        f'train_loss: {train_loss:.5f} ' +
                        f'valid_loss: {valid_loss:.5f}'
        )
        print(print_msg)

        # clear loss list
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    torch.cuda.empty_cache()