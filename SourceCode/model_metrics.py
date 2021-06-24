import sys, os
import numpy as np
from PIL import Image
from torch import nn, optim, cuda, no_grad, exp, save, load, Tensor, flatten
import torch.nn.functional as F

def recall():
    # Check that saved model exists
    if not os.path.isfile('mnist_model.pt'):
        return

    else:
        # print("Recall per digit:")
        recalls = []
        for i in range(0, 10):
            img_id = 0
            predictions = []
            correct = 0

            for j in range (0, len(os.listdir(f'metrics_images/{i}'))):
                # Open the image and convert it to a tensor, values ranged from 0-1
                user_input = Image.open(f'metrics_images/{i}/i{img_id}.png').convert('L')
                user_input = 1 - (np.array(user_input)/255)
                user_input = Tensor(user_input.reshape(-1, 1, 28, 28))

                # Load the saved trained model
                model = load('mnist_model.pt')

                # Feed image into trained model
                with no_grad():
                    ps = model(user_input)
                probs = list(ps.numpy()[0])
                
                # Scales the probabilities to decimals between 0 and 1
                tot_probs = 0
                for k in range(len(probs)): # Gets sum of all probabilities
                    tot_probs += probs[k]
                scaled_probs = probs/tot_probs # Divide each probability by the total
                img_id += 1

                predictions.append(probs.index(max(probs)))
                # print(f'{j}: {probs.index(max(probs))}')

            for j in range(0, len(predictions)):
                if predictions[j] == i:
                    correct += 1
            recalls.append(correct/len(predictions)*100)
            # print(f'{i}: {round(correct/len(predictions)*100, 3)}%')

        tot_recall = 0
        for i in range(0,10):
            tot_recall += recalls[i]
        
        return tot_recall/10


def precision():
    # Check that saved model exists
    if not os.path.isfile('mnist_model.pt'):
        return

    else:
        # print("Precision per digit:")
        correct_ratio = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        for i in range(0, 10):
            img_id = 0

            for j in range (0, len(os.listdir(f'metrics_images/{i}'))):
                # Open the image and convert it to a tensor, values ranged from 0-1
                user_input = Image.open(f'metrics_images/{i}/i{img_id}.png').convert('L')
                user_input = 1 - (np.array(user_input)/255)
                user_input = Tensor(user_input.reshape(-1, 1, 28, 28))

                # Load the saved trained model
                model = load('mnist_model.pt')

                # Feed image into trained model
                with no_grad():
                    ps = model(user_input)
                probs = list(ps.numpy()[0])
                
                # Scales the probabilities to decimals between 0 and 1
                tot_probs = 0
                for k in range(len(probs)): # Gets sum of all probabilities
                    tot_probs += probs[k]
                scaled_probs = probs/tot_probs # Divide each probability by the total
                img_id += 1

                correct_ratio[probs.index(max(probs))][0] += 1
                if probs.index(max(probs)) == i:
                    correct_ratio[probs.index(max(probs))][1] += 1

            # for j in range(0, len(predictions)):
            #     if predictions[j] == i:
            #         correct += 1
            # correct_ratio.append(correct/len(predictions)*100)
            # print(f'{i}: {round(correct/len(predictions)*100, 3)}%')
        predictions = []
        for i in range(0, 10):
            try:
                predicted = correct_ratio[i][0]
                correct = correct_ratio[i][1]
                predictions.append((correct/predicted)*100)
            except:
                pass
            # print(f'{i}: {round(predictions[i], 3)}')

        tot_precision = 0
        for i in range(0,len(predictions)):
            tot_precision += predictions[i]
        
        return tot_precision/len(predictions)
        
def f1(recall, precision):
    return 2*((precision*recall)/(precision+recall))

 
def main():
    model_precision = precision()
    model_recall = recall()
    model_f1 = f1(model_recall, model_precision)

    print(f'Model precision: {round(model_precision, 3)}%\n'\
        f'Model recall: {round(model_recall, 3)}%\n'\
        f'Model F1: {round(model_f1, 3)}%')


#--------------------LeNet5-----------------------#
class LeNet5(nn.Module): 

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Sequential(
            # 3 convolution layers
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4, stride=1),
            nn.Tanh()
        )
        self.fc = nn.Sequential(
            # Two fully connected layers 
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10)
        )


    def forward(self, x):
        
        x = self.conv1(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return F.softmax(x, dim=1)
#-------------------------------------------------#



#---------------LINEAR REGRESSION-----------------#

class LinReg(nn.Module):
    def __init__(self):
        super(LinReg, self).__init__()
        # One linear transformation
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        x = flatten(x, 1)
        x = self.linear(x)
        return exp(x)

#-------------------------------------------------#



#--------------MULTILAYER PERCEPTRON--------------#

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            # A linear transformation, activation layer then an output layer
            nn.Linear(784, 120),
            nn.ReLU(),
            nn.Linear(120, 10)
        )

    def forward(self, x):
        x = flatten(x, 1)
        x = self.layers(x)
        return exp(x)

#-------------------------------------------------#


main()