from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils.config_utils import read_args, load_config, Dict2Object
import random
import numpy
import torch.multiprocessing as mp
import time
import statistics
from multiprocessing import  Manager
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def accuracy_fn(a,b):
    correct = torch.eq(a, b).sum().item()
    acc = ( correct / len(b) ) * 100
    return acc


def train(args, model, device, train_loader, optimizer, epoch, accuracy_fn,seed):
    """
    train the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    running_loss=0.0
    running_acc=0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''Fill your code...'''
        running_loss += loss.item()
        running_acc += accuracy_fn(a=target, b=output.argmax(dim=1))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch:{}[{}/{} ({:.0f}%)]\t Loss : {:.6f}\t Accuracy :{:.6f}' .format(
            epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item(),running_acc/len(train_loader)))

    training_acc, training_loss = running_acc/len(train_loader), running_loss/len(train_loader) # replace this line
    with open("Train.txt","a") as f:
        f.write("\n")
        f.write(f"Epoch: "+ str(epoch) + " | Training Loss: "+ str(training_loss)+ " | Training Acc: "+str(training_acc)+" | seed:" + str(seed))
    print(f"Epoch: {epoch} | Loss: {training_loss:.5f}, Acc: {training_acc:.2f}% ")
    return training_acc, training_loss


def test(model, device, test_loader,epoch,seed):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.inference_mode():
        for data, target in test_loader:
            '''Fill your code...'''
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            test_loss += loss.item()
            test_acc += accuracy_fn(a = target, b = output.argmax(dim=1))
    testing_acc, testing_loss = test_acc/len(test_loader), test_loss/len(test_loader)  # replace this line
    with open("Test.txt", "a") as f:
        f.write("\n")
        f.write(f"Epoch: " + str(epoch) + " | Test Loss: " + str(testing_loss) + " | Test Acc: " + str(testing_acc) +" | seed:" + str(seed))
    print(f'Epoch: {epoch} | Test Loss: {testing_loss:.5f} | Test Acc: {testing_acc:.2f}% ')
    return testing_acc, testing_loss


def plot(epoches, performance, name, seed):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    plt.xlabel("Epoches")
    plt.ylabel(name)
    plt.title("Epoches-" + name+" function || seed: "+str(seed))
    plt.plot(epoches, performance)
    plt.show()
    pass

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(0)

def run(config,seed,training_accuracies0,training_loss0,testing_accuracies0,testing_losses0):
    """ 选择运行硬件"""
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(seed)
    if use_cuda:
        device = torch.device("cuda")
        print("cuda")
    elif use_mps:
        device = torch.device("mps")
        print("mps")
    else:
        device = torch.device("cpu")
        print("cpu")
    train_kwargs = {'batch_size': config.batch_size}
    test_kwargs = {'batch_size': config.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    """add random seed to the DataLoader, pls modify this function"""
    #数据库的读取器 Dataloader
    #添加random seed
    train_loader = torch.utils.data.DataLoader(dataset1,batch_size=config.batch_size,worker_init_fn=seed_worker,generator=g)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=config.batch_size,worker_init_fn=seed_worker,generator=g)
    model = Net().to(device)
    model.share_memory()
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)
    """record the performance"""

    epoches = []
    training_accuracies = []
    training_loss = []

    testing_accuracies = []
    testing_losses = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch,accuracy_fn,seed)
        """record training info, Fill your code"""
        epoches.append(epoch)
        train_loss = train_loss\
            #.detach().numpy()
        training_loss.append(train_loss)
        training_accuracies.append(train_acc)
###
        test_acc, testing_loss = test(model, device, test_loader,epoch,seed)
        """record testing info, Fill your code"""
        testing_loss = testing_loss\
            #.detach().numpy()
        testing_losses.append(testing_loss)
        testing_accuracies.append(test_acc)
###
        scheduler.step()
        """update the records, Fill your code"""
    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    """plotting training performance with the records"""
    name1 = "training_loss"
    name2 = "training_accuracies"
    name3 = "testing_accuracies"
    name4 = "testing_losses"
    plot(epoches= epoches, performance=training_loss, name=name1,seed=seed)
    plot(epoches= epoches,performance=training_accuracies, name=name2,seed=seed)
    """plotting testing performance with the records"""
    plot(epoches= epoches,performance=testing_accuracies, name=name3,seed=seed)
    plot(epoches= epoches,performance=testing_losses, name=name4,seed=seed)
    training_accuracies0.append(training_accuracies)
    training_loss0.append(training_loss)
    testing_accuracies0.append(testing_accuracies)
    testing_losses0.append(testing_losses)

def plot_mean(meanvalue,epochs,name):
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    """fill your code"""
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(str(name) + " with different seeds")
    plt.plot(epochs, meanvalue)
    plt.show()
    pass

if __name__ == '__main__':
    manger=Manager()
    training_accuracies0 = manger.list()
    training_loss0 = manger.list()
    testing_accuracies0 = manger.list()
    testing_losses0 = manger.list()

    arg = read_args()
    """toad training settings"""
    config = load_config(arg)
    """train model and record results"""
    p1 = mp.Process(target=run,args=(config,123,training_accuracies0,training_loss0,testing_accuracies0,testing_losses0))
    p2 = mp.Process(target=run,args=(config,321,training_accuracies0,training_loss0,testing_accuracies0,testing_losses0))
    p3 = mp.Process(target=run,args=(config,666,training_accuracies0,training_loss0,testing_accuracies0,testing_losses0))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    mean_test_acc =[0.0]*config.epochs
    mean_test_loss =[0.0]*config.epochs
    mean_train_acc =[0.0]*config.epochs
    mean_train_loss =[0.0]*config.epochs
    epochs=[]
    for epoch in range(1,config.epochs+1):
        epochs.append(epoch)
    for i in range(0,3):
        for j in range(0, config.epochs):
            mean_train_acc[j] +=(float)(training_accuracies0[i][j]/3)
            mean_train_loss[j] +=(float)(training_loss0[i][j]/3)
            mean_test_acc[j] +=(float)(testing_accuracies0[i][j]/3)
            mean_test_loss[j] +=(float)(testing_losses0[i][j]/3)
    name1 = "mean_test_acc"
    name2 = "mean_test_loss"
    name3 = "mean_train_acc"
    name4 = "mean_train_loss"

    """plot the mean results"""
    plot_mean(mean_train_acc, epochs, name3)
    plot_mean(mean_train_loss, epochs, name4)
    plot_mean(mean_test_acc, epochs, name1)
    plot_mean(mean_test_loss, epochs, name2)
