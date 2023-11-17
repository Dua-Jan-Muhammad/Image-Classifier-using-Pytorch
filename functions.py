import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image

def loading_transforming_data(image_data_dir):
    '''
    Transforms the data and returns imageFolders and Dataloaders
 
    Args:
        data_dir (string): path where the data is.
   
    Returns:
       trainloader (torch.utils.data.dataloader.DataLoader): Image data for Training.
       validloader (torch.utils.data.dataloader.DataLoader): Image data for Validation.
       testloader (torch.utils.data.dataloader.DataLoader): Image data for Testing.
       train_data (torchvision.datasets.folder.ImageFolder): Images Data to pass to DataLoder.
       valid_data (torchvision.datasets.folder.ImageFolder): Images Data to pass to DataLoder.
       test_data (torchvision.datasets.folder.ImageFolder): Images Data to pass to DataLoder.
    '''
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) 

    valid_transforms= transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]) 


    train_data = datasets.ImageFolder(image_data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(image_data_dir + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(image_data_dir + '/test', transform=valid_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return trainloader, validloader, testloader, train_data, valid_data, test_data


def model_Classifier(model, input_units, hidden_units, dropout):
     '''
    takes the pretrained model, freezes parameters, build a classifier, assign a classifier to that         model and returns the model with updated classifier function
 
    Args:
        model (torchvision.models.modelName): models.resnet50(pretrained=True).
        input_units (int): No of initial or input units.
        hidden_units (int): No of Hidden units that need to be updated in a classifier.
        dropout (float): dropout rate 
        
   
    Returns:
       model (torchvision.models.modelName): model which parameters are freeze and classifier has been updated according to given arguments.
    '''
     for param in model.parameters():
        param.requires_grad = False  #freezing parameter
    
     Classifier =  nn.Sequential(
         nn.Linear(input_units, hidden_units),
         nn.ReLU(),
         nn.Dropout(dropout),
         nn.Linear(hidden_units,102),
         nn.LogSoftmax(dim=1)
    )

     model.fc = Classifier
     return model


def training_model(epochs,trainloader, validloader, GPU_mode, model, criterion, optimizer):
    '''
    trains the model
 
    Args:
        epochs (int): no of epochs.
        trainloader (torch.utils.data.dataloader.DataLoader): Image data for Training.
        validloader (torch.utils.data.dataloader.DataLoader): Image data for Validation.
        GPU_mode (Boolean): If true, data and model will be shifted to CuDA(GPU) for training.
        model (torchvision.models.modelName): model which was updated with new classifier.
        criterion (torch.nn.modules.loss.lossfunctionName): loss function from Pytorch.
        optimizer (torch.optim.OptimizerName): Optimizer function from Pytorch.
   
    Returns:
       model (torchvision.models.modelName): Trained model.
       optimizer (torch.optim.OptimizerName): updated Optimizer function.
    '''
    device = torch.device("cuda" if GPU_mode else "cpu")
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 10
    print('epoch training has started')


    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            #this lines make sure that the gradients are recalculated from the start each new time
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"valid loss: {valid_loss/len(validloader):.3f}.. "
                    f"valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                return model, optimizer
     
def testing_model(model, testloader,GPU_mode ):
    '''
    Test the model on testloader and Prints the accuracy of the model.
 
    Args:
        model (torchvision.models.modelName): Trained model that was returned using training_model functiion.
        testloader(torch.utils.data.dataloader.DataLoader): as was returned in loading_transforming_data function.
        GPU_mode(Boolean): if true, the model and data would be sent to GPU mode.
   
    Returns:
       None
    '''
    
    device = torch.device("cuda" if GPU_mode else "cpu")
    model.to(device)
    
    test_loss = 0
    accuracy = 0
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)

            _, predicted_out = torch.max(outputs.data, 1)            
            test_loss += labels.size(0)                               
            accuracy += (predicted_out == labels).sum().item()
            
    print(f"Accuracy of this model is: {round(100 * accuracy / test_loss,3)}%")
    
    
    
def saving_checkpoint(model, train_data, epoch, optimizer, checkpoint_path ):
    '''
    Saves the current model to the .pth file on the given location.
 
    Args:
       model (torchvision.models.modelName): Trained model that was returned using training_model functiion.
        train_data (torchvision.datasets.folder.ImageFolder): as was returned in loading_transforming_data function.
        dir_path(string): where the model would be saved.
        ptimizer (torch.optim.OptimizerName): updated Optimizer function as was returned in training_model function
        epoch(int): no of epochs.
   
    Returns:
       None
    '''
    
    checkpoint = {
        'classifier': model.fc,
        'class_to_idx': train_data.class_to_idx,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    
def load_model(dir_path,GPU_mode, model):
    '''
    Loads the Saved the model from .pth file path.
 
    Args:
        dir_path(string): From where the model would be loaded.
        GPU_mode(Boolean): if true, the model and data would be sent to GPU mode.
        model (torchvision.models.modelName): model where this checkpoint would be updated.
   
    Returns:
       None
    '''
    dir_path = str(dir_path)
    checkpoint = torch.load(dir_path)
    device = torch.device("cuda" if GPU_mode else "cpu")
    model.to(device)
   
    model.fc = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epoch']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array.
    Args:
        image(string): Path of image.
  
    Returns:
       image(numpy.ndarray): numpy array of an image.
    '''
    #open and image
    img = Image.open(image)

    #resize image
    img.thumbnail((256,256))

    img_width, img_height = img.size

    #center crop image
    img2 = img.crop(((img_width - 224) / 2,
                         (img_height - 224) / 2,
                         (img_width + 224) / 2,
                         (img_height + 224) / 2))

    # convert to np array
    np_image = np.array(img2)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    np_image = np_image / 255

    # normalizing with respect ot mean and std
    normailzed_image = np_image - mean / std

    # taking transpose
    image = np_image.transpose((2,1,0))

    return image

def predict(image_path, model, GPU_mode,topk=5, ):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    Args:
        image(string): Path of image.
        model (torchvision.models.modelName): Trained model.
        GPU_mode(Boolean): if true, the model and data would be sent to GPU mode.
        
  
    Returns:
       top_probabilities (list): top 5 probabilities.
       top_classes (list): Classes name of that top 5 probabilities.
    '''

    #sending image path and getting numpy array
    img = process_image(image_path)

    # converting array to tensor
    image_tensor = torch.from_numpy(img).float()
    image_tensor = image_tensor.unsqueeze(0)
  
    device = torch.device("cuda" if GPU_mode else "cpu")
    
    Image_tensor_on_GPU = image_tensor.to(device);
   
    model.eval()

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(Image_tensor_on_GPU)
        ps = torch.exp(output)
        top_probabilities, top_indices = ps.topk(topk)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Map the indices to class labels
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    
    return top_probabilities[0].tolist(), top_classes