import argparse
import sys

import torch
import click
from torch import nn, optim

from data import mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt
import numpy as np


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    trainloader, _ = mnist()
    #images = train_set['images']
    #labels = train_set['labels']

    criterion =  nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr)
    num_epochs = 30

    model.train()
    train_losses = []
    epochs = []
    for epoch in range(num_epochs):
        running_loss = 0
        
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_losses.append(running_loss/len(trainloader))
            epochs.append(epoch)
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")
    
    print('Finished Training Trainset')
    torch.save(model.state_dict(), 'trained_model.pth')        
    plt.plot(np.array(epochs,train_losses), 'r')
    plt.show()
    #plt.savefig('loss.png')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)
    
    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    _, testloader = mnist()
    
    acc=[]

    with torch.no_grad():

        model.eval()

        for images, labels in testloader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            acc.append(accuracy.item()*100)
            #print(f'Accuracy: {accuracy.item()*100}%')
        print(f'Final Accuracy: {sum(acc)/len(acc)}%')



cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    
