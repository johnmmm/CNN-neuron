from network import Network
from layers import Relu, Linear, Conv2D, AvgPool2D, Reshape
from utils import LOG_INFO
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_4d
from utils import writer

train_data, test_data, train_label, test_label = load_mnist_4d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Conv2D('conv1', 1, 4, 3, 1, 0.01))
model.add(Relu('relu1'))
model.add(AvgPool2D('pool1', 4, 0))  # output shape: N x 4 x 14 x 14
# model.add(Conv2D('conv2', 4, 4, 3, 1, 0.01))
# model.add(Relu('relu2'))
# model.add(AvgPool2D('pool2', 2, 0))  # output shape: N x 4 x 7 x 7
model.add(Reshape('flatten', (-1, 196)))
model.add(Linear('fc3', 196, 10, 0.1))

loss = SoftmaxCrossEntropyLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 201,
    'disp_freq': 10,
    'test_epoch': 3
}

writer.writerow(['type','middle_layers','learning_rate','weight_decay','momentum'])
rlist=[]
rlist.append('Relu')
rlist.append('buzhidao')
rlist.append(config['learning_rate'])
rlist.append(config['weight_decay'])
rlist.append(config['momentum'])
writer.writerow(rlist)
writer.writerow(['iter_counter','loss_list','acc_list'])

for epoch in range(config['max_epoch']):
    # config['learning_rate'] = 0.02 - (epoch / 200) * (0.02 - 0.001)
    # print(epoch / 200)
    # print(config['learning_rate'])

    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        test_net(model, loss, test_data, test_label, config['batch_size'])
