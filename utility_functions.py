# Imports here
# PyTorch
from torchvision import transforms #to do transformations
from torchvision import datasets #to create a datasets 
from torch.utils.data import DataLoader # to create a dataloder DataLoaders 
from torchvision import models  # Loading in a pre-trained model in PyTorch VGG-16
import torch
from torch import optim, cuda
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict # to create the sequential model
# Data science tools
import numpy as np
import os # to open the files 
import pandas as pd # to create the history of loss and accuracy dataframe

# Image manipulations
from PIL import Image

# Timing utility
from timeit import default_timer as timer # to calculate the spended time

# Visualizations
import matplotlib.pyplot as plt # to plot the images 
plt.rcParams['font.size'] = 14

import json # to load the json file

# Define the global parameter 
batch_size = 64

#Define different directories of the data 
def data_dirs(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_dir = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
    return data_dir

# Done: Define your transforms for the training, validation, and testing sets
def set_loaders(data_dir):
    # Done: Define your transforms for the training, validation, and testing sets
# Image transformations
    data_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])  # Imagenet standards
        ]),
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Done: Load the datasets with ImageFolder
    image_datasets =  {x: datasets.ImageFolder(data_dir[x],
                                              data_transforms[x])
                      for x in ['train', 'valid','test']}

    # Done: Using the image datasets and the trainforms, define the dataloaders
    # Data iterators
    dataloaders ={x: DataLoader(image_datasets[x], batch_size = batch_size,
                                                 shuffle=True)
                  for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders 


#Loading in a pre-trained model
def load_pretrained_model(arch):
    # Init the diff models 
    vgg16 = models.vgg16(pretrained=True)
    densenet161 = models.densenet161(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    # dictionary of the diff models
    _models = {'vgg': vgg16, 'densenet': densenet161, 'alexnet': alexnet}
    model = _models['vgg'] # if no architect selected we set the architect as vgg
    # If arch in the dict set the model with the given arch
    if arch in _models:
        model = _models[arch]
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    return model 

# Build the classifier
def build_classifier(model, arch,hidden_units, dataloaders, drop_prob=0.5):
    # freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    if (arch=='densenet'):
         n_inputs = model.classifier.in_features # get the number of inputs 
    else: 
         n_inputs = model.classifier[-1].in_features # get the number of inputs 

    n_classes = len(dataloaders['train'].dataset.classes) # get the number of classes => the number of categories of train dataset
  
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(n_inputs, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(p=drop_prob)),
        ('fc2', nn.Linear(int(hidden_units), n_classes)),
        ('out', nn.LogSoftmax(dim=1))
    ]))
    if (arch=='densenet'):
         model.classifier = classifier 
    else:
        model.classifier[-1] = classifier # replace the last layer with our custom model
    print('The model.classifier => \n {} \n'.format(model.classifier))
    return model


# Implement a function for the validation pass. Ps we can use it also to test the model 
def validation(model, valid_loader, criterion, train_on_gpu):
    # Make sure network is in eval mode for inference
    model.eval()
    valid_loss = 0
    valid_acc = 0
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad(): 
        for images, labels in valid_loader:
             # Tensors to gpu
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            # Forward pass    
            output = model(images)
            # Validation loss
            loss = criterion(output, labels)
            # Multiply average loss times the number of examples in batch
            valid_loss += loss.item() * images.size(0)

            # Calculate validation accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            valid_acc += accuracy.item() * images.size(0)  
    # Calculate mean losses
    valid_loss = valid_loss / len(valid_loader.dataset)
    # Calculate mean accuracy
    valid_acc = valid_acc / len(valid_loader.dataset)  
    return valid_loss, valid_acc

# Function to train the model 
def train_model(model,
          train_on_gpu,
          learning_rate,      
          train_loader,
          valid_loader,
          save_file_name,
          epochs,
          epochs_stop=5,
          print_every=4):
    '''
        The role of this function is to train the model 
        
        Params: 
            model: Pytorch model, the model to train 
            train_on_gpu: boolean, wheter to train or not on gpu device
            learning_rate: float, the learning rate of the optimizer
            train_loader: PyTorch dataloader, the training data loader to iterate through
            valid_loader: PyTorch dataloader, the dataset used for validation 
            save_file_name: String ending with '.pt' extenstion, the file to save the model state dictionary 
            epoches_stop: int, the maximun number of epoches to stop from after no improvement 
            epoches: int, the number of epoches for training 
            print_every: int, frequency of epochs to print training stats
                
        Returns: 
            model: Pytorch model, the trained model 
            history: DataFrame, history of train and validation loss & accuracy
    '''
    # Move the model to gpu 
    if train_on_gpu:
        model = model.to('cuda')
    
    #define the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=learning_rate)

    history = [] # to store the history here 
    valid_max_acc = 0
    # Early stopping Initialization
    epochs_without_improve = 0
    valid_loss_min = np.Inf
    model.epochs = 0
    print('Start Training... \n')

    overall_start = timer() # to calculate the overall time
    for epoch in range(epochs):
        # Keep track of training & validation loss and accuracy
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0 
        valid_acc = 0 
        model.train()
        start = timer() # to caculate the consumed time for every epoch
        #for images, labels in iter(train_loader):
        for ii, (images, labels) in enumerate(train_loader):
            # move to gpu 
            if train_on_gpu:
                images, labels = images.cuda(), labels.cuda()
            # Clear the gradients, do this because gradients are accumulated
            optimizer.zero_grad()
            # Forward pass, then backward pass, then update weights
            output = model(images)
            # Loss and backpropagation of gradients
            loss = criterion(output, labels)
            loss.backward()
            # Update the parameters
            optimizer.step()
            # Track train loss
            train_loss += loss.item() * images.size(0)         
            # Calculate the accuracy
            _, pred = torch.max(output, dim = 1)
            correct_tensor = pred.eq(labels.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * images.size(0) 
            # Track the training
            #print('Epoch: {}\t {:.2f}% completed \t Elapsed seconds: {:.2f}'.format(epoch, 100 * ii / len(train_loader), timer() - start))
        # Here we use the else block just after for/while loop to execute it only 
        #hen the loop is NOT terminated by a break statement
        else:
            # Increase the number of epochs after each statement
            model.epochs += 1         
            # Make sure network is in eval mode for inference
            model.eval()
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad(): 
                ## Run validation after every epoch :
                valid_loss, valid_acc = validation(model, valid_loader, criterion, train_on_gpu)
                # Calculate mean losses
                train_loss = train_loss / len(train_loader.dataset)
                # Calculate mean accuracy
                train_acc = train_acc / len(train_loader.dataset)           
                history.append([train_loss, valid_loss, train_acc, valid_acc])          
                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(train_loss),
                      "Valid Loss: {:.3f}.. ".format(valid_loss),
                      "Train Accuracy: {:.2f}%".format(100 * train_acc),
                      "Valid Accuracy: {:.2f}%".format( 100 * valid_acc))               
                # Early stopping based on the max number of epoches witout improvement
                # Save the model if the validation and the loss decreases
                if valid_loss < valid_loss_min - 0.01:
                    print('Validation loss decreased ({:.4f} => {:.4f}).  Saving model ...'.format(valid_loss_min, valid_loss))                         # Save the model
                    torch.save(model.state_dict(), save_file_name)
                    epochs_without_improve = 0
                    valid_loss_min = valid_loss
                    best_epoch = epoch
                # Else increment the number of epoches without improvement
                else:
                    epochs_without_improve += 1
                    print('Number of epoches without improvement: {}'.format(epochs_without_improve))
                    if epochs_without_improve >= epochs_stop:
                        print('Early Stopping')
                        total_time = timer() - overall_start
                        print('Total time: {:.2f} seconds.Time/epoch: {:.2f} seconds'.format(total_time,
                                                                                     total_time / (epoch+1)))
                        # Load the best state dictionary
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer
                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history                   
    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print('The best epoch: {} with loss {:.2f} and accuracy: {:.2f}%'.format(best_epoch, valid_loss_min,100* valid_acc))
    if(epoch != 0):
        print("Total time: {:.2f} seconds. Time per epoche was {:.2f} seconds".format(total_time,
                                                                          total_time / epoch ))
    
    # Format the history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history       

# Function to test the model 
def test_model(model, testloader, train_on_gpu):
    #define the criterion 
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validation(model, testloader, criterion, train_on_gpu)
    #Print the result 
    print("Test Loss: {:.3f}.. ".format(test_loss),"Test Accuracy: {:.2f}%".format( 100 * test_acc))
   

def cat_to_name():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name
#Mapping classes to Indexes
def idx_to_name(train_set): 
    class_to_idx = train_set.class_to_idx
    cat_name = cat_to_name()
    idx_to_name = {
        idx: cat_name[class_]
        for class_, idx in train_set.class_to_idx.items()
    }
    return idx_to_name

# Done: Save the checkpoint 
def save_checkpoint(model,epochs ,learning_rate, path, train_set):
    print('Saving the model to ./{}/checkpoint.pth'.format(path), flush = True)
    checkpoint = {
        'model': model,
        'cat_to_name': cat_to_name(),
        'class_to_idx': train_set.class_to_idx,
        'idx_to_name': idx_to_name(train_set),
        'epochs': epochs,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict()
    }
    
    if not os.path.exists(path):
        print('save directories...', flush = True)
        os.makedirs(path)
    torch.save(checkpoint, path + '/checkpoint.pth')

# Load the checkpoint and rebuild the model 
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    # Load the saved file    
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  
    # Rebuild the model
    model.classifier = checkpoint['classifier']
    model.cat_to_name = checkpoint['cat_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_name = checkpoint['idx_to_name']
    model.epochs = checkpoint['epochs']
    # Load the state dict with torch.load 
    model.load_state_dict(checkpoint['state_dict'])
    return model


 # Done: Process a PIL image for use in a PyTorch model
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image_path)
    # Resize the images where the shortest side is 256 pixels using resize method 
    img = image.resize((256, 256))  
    # Crop out the center 224x224 portion of the image using crop method
    width = 256 
    height = 256
    new_width = 224
    new_height = 224
    # Setting the points for cropped image 
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2   
    # Cropped image of above dimension 
    img = img.crop((left, top, right, bottom))      
    # Convert to numpy
    img = np.array(img)
    # Reorder the dimension of the color to the first position using ndarray.transpose and retain the order of the other dimension
    img = img.transpose((2, 0, 1)) 
    # Normalize the image 
    img = img / 256  
    # Standardize the Image 
    # Define the mean which is [0.485, 0.456, 0.406] 
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    # Define the standar deviation which is  [0.229, 0.224, 0.225]
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    # Subtract the means from each color channel, then divide by the standard deviation
    img = (img - means) / stds
    #Convert the image to pytorch tensor
    img_tensor = torch.Tensor(img)
    
    return img_tensor


# Done: Implement the code to predict the class from an image file
def predict(image, model, train_on_gpu,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''          
    # Resize the image tensore 
    if train_on_gpu:
        model = model.to('cuda')
        image = image.view(1, 3, 224, 224).cuda()
    else:
        image = image.view(1, 3, 224, 224)
    # Evaluation
    with torch.no_grad():    
        model.eval()
        # Model outputs log probabilities
        out = model(image)   
        ps = torch.exp(out)
        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)
        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_name[category] for category in topclass.cpu().numpy()[0]
        ]
        top_prob = topk.cpu().numpy()[0]
        
        return top_prob, top_classes, topclass.cpu().numpy()[0]

   
