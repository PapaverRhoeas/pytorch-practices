from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import numpy as np
import pandas as pd
import visdom
import copy
import os
import misc
import model
import dataset
from tensorboardX import SummaryWriter

# hyperparameters during training procedure, other parameters also can change
name = 'LeNet_tensorboardX'
batch_size = 32
num_epochs = 10
learning_rate = 0.01
lr_scheduler = [5]  # default gamma is 0.1

# prepare to print log file in 'log' folder
logdir = 'log'
logdir = os.path.join(os.path.dirname(__file__), logdir)
print(logdir)
misc.ensure_dir(logdir)
misc.logger.init(logdir, '{}_log'.format(name))
print = misc.logger.info

# activate a visualize window
vis = visdom.Visdom(env='MNIST tutorial')

# print out the system condition
print("PyTorch Version: {}".format(torch.__version__))
print("Torchvision Version: {}".format(torchvision.__version__))
print('CUDA available: {}'.format(torch.cuda.is_available()))
if torch.cuda.is_available():
    x = torch.Tensor([1.0])
    xx = x.cuda()
    print('cuDNN available: {}'.format(torch.backends.cudnn.is_acceptable(xx)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Model is running in {}'.format(device))

# Print out the HyperParameters
print('HyperParameters :\n- name={}\n- batch_size={}\n- num_epochs={}\n- learning_rate={}\n- lr_scheduler={}\n'.
      format(name, batch_size, num_epochs, learning_rate, lr_scheduler))


# define the train procedure
def train_model(model, ceriterion, optimizer, scheduler, train_epochs):

    writer = SummaryWriter('./log_tbX')

    val_acc_history = []
    tra_acc_history = []
    val_loss_history = []
    tra_loss_history = []
    x_epochs = []
    train_status = vis.text('Start Training',
                            opts=dict(title='Training Status'))
    vis.text('HyperParameters : name={}, batch_size={}, num_epochs={}, learning_rate={}, lr_scheduler={}'.
             format(name, batch_size, num_epochs, learning_rate, lr_scheduler), win=train_status, append=True)

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0
    time_total = 0
    for epoch in range(train_epochs):
        since = time.time()
        print('Epoch {}/{}'.format((epoch+1), num_epochs))
        vis.text('Epoch {}/{}'.format((epoch + 1), num_epochs), win=train_status, append=True)
        print('-' * 20)
        vis.text(('-' * 20), win=train_status, append=True)
        x_epochs.append(epoch + 1)

        running_loss = 0
        running_correct = 0
        scheduler.step()
        print('Current LR is %.5f' % optimizer.param_groups[0]['lr'])
        vis.text('Current LR is {:.5f}'.format(optimizer.param_groups[0]['lr']), win=train_status,
                 append=True)
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()

            if i % 20 == 0:
                niter = epoch * len(trainloader) + i
                writer.add_scalar('Train_Loss', loss.item() * inputs.size(0), niter)
                for tag, value in model.named_parameters():
                    writer.add_histogram(tag=tag, values=value, global_step=niter)

        tra_loss_history.append(running_loss/len(trainset))
        tra_acc_history.append(running_correct/len(trainset))

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(tra_loss_history[-1], tra_acc_history[-1]))
        vis.text('Train Loss: {:.4f} Acc: {:.4f}'.format(tra_loss_history[-1], tra_acc_history[-1]),
                 win=train_status, append=True)

        running_loss = 0
        running_correct = 0

        with torch.no_grad():
            for i, data in enumerate(testloader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                running_correct += (predicted == labels).sum().item()

                if i % 10 == 0:
                    niter = epoch * len(testloader) + i
                    writer.add_scalar('Val_Loss', loss.item() * inputs.size(0), niter)

            val_loss_history.append(running_loss / len(testset))
            val_acc_history.append(running_correct / len(testset))

            if val_acc_history[-1] > best_acc:
                best_acc = val_acc_history[-1]
                best_model_weights = copy.deepcopy(model.state_dict())

        print('Validate Loss: {:.4f} Acc: {:.4f}'.format(val_loss_history[-1], val_acc_history[-1]))
        vis.text('Validate Loss: {:.4f} Acc: {:.4f}'.format(val_loss_history[-1], val_acc_history[-1]),
                 win=train_status, append=True)

        writer.add_scalars('./Loss', {'Train_loss': tra_loss_history[-1],
                                      'Val_loss': val_loss_history[-1]}, epoch)
        writer.add_scalars('./Acc', {'Train_acc': tra_acc_history[-1],
                                     'Val_acc': val_acc_history[-1]}, epoch)

        if epoch + 1 > 1:
            vis.close(win=Loss)
            vis.close(win=Acc)
        Loss = vis.line(
            Y=np.column_stack((tra_loss_history, val_loss_history)),
            X=np.column_stack((x_epochs, x_epochs)),
            opts=dict(
                title=('Loss after {} epoch'.format(epoch + 1)),
                legend=['Tra_loss', 'Val_loss'],
                showlegend=True,
            ), )
        Acc = vis.line(
            Y=np.column_stack((tra_acc_history, val_acc_history)),
            X=np.column_stack((x_epochs, x_epochs)),
            opts=dict(
                title=('Acc after {} epoch'.format(epoch + 1)),
                legend=['Tra_Acc', 'Val_Acc'],
                showlegend=True,
            ), )

        time_elapsed = time.time() - since
        time_total += time_elapsed
        print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        vis.text('Complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), win=train_status, append=True)

    print('All complete in {:.0f}m {:.0f}s'.format(time_total // 60, time_total % 60))
    vis.text('All complete in {:.0f}m {:.0f}s'.format(
            time_total // 60, time_total % 60), win=train_status, append=True)

    torch.save(model.state_dict(), './models/{}_final_weights.pkl'.format(name))
    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), './models/{}_best_weights.pkl'.format(name))

    writer.close()
    return tra_loss_history, tra_acc_history, val_loss_history, val_acc_history


# load the datesets
trainset, trainloader, testset, testloader = dataset.getdata(batch_size=batch_size)
print(trainset)
print(testset)

# load the LeNet model
LeNet_model = model.make_model().to(device)
print(LeNet_model)

# load the optimizer tools
criterion = nn.CrossEntropyLoss()  # combines logsoftmax and NLLLoss
optimizer = optim.SGD(LeNet_model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_scheduler, gamma=0.1)

# start training procedure, already save the best and final model to 'models' folder
Tra_loss, Tra_acc, Val_loss, Val_acc = train_model(LeNet_model, criterion, optimizer, scheduler, num_epochs)

# save the data of loss and acc to 'results' folder
Loss_save = np.vstack((Tra_loss, Val_loss))
data1 = pd.DataFrame(data=Loss_save, index=['Tra_loss', 'Val_loss'])
data1 = data1.T
data1.to_csv('./results/{}_Loss.csv'.format(name))
Acc_save = np.vstack((Tra_acc, Val_acc))
data2 = pd.DataFrame(data=Acc_save, index=['Tar_acc', 'Val_acc'])
data2 = data2.T
data2.to_csv('./results/{}_Acc.csv'.format(name))

# Handwrite test.py is used to test the performance of trained model using some handwritten pictures
