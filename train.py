import argparse
import functions
from torchvision import datasets, transforms, models
from torch import nn, optim

def main():
    parser = argparse.ArgumentParser( description='arguments to run the train script')
    
    
    parser.add_argument('--image_data_dir', action='store',
                        required=True,
                        default=None,
                        help='Enter path to images data. eg: "./Flowers" ')
    
    parser.add_argument('--PTmodel', 
                        dest='pretrained_model', default='resnet50',
                        choices=['inception_v3','resnet50', 'resnet18', 'resnet34','resnet101'],
                        help='Enter model to use. eg:resnet50 or inception_v3.  this classifier can currently work with\ resnet architectures.')

    parser.add_argument('--checkpoint', 
                        dest='checkpoint', default='checkpoint.pth',
                        help='Enter location to save checkpoint in.')
    
    parser.add_argument('--hidden_units', dest='hidden_units', default = 1000,
                       help='Enter No of hidden units in a classifier')

    parser.add_argument('--epochs', dest='epochs', default = 5,
                         help='Enter No of epochs to run')
       
    
    parser.add_argument('--learning_rate', dest='learning_rate', default = 0.003,
                       help='Enter a Learning Rate')
        
    parser.add_argument('--dropout_rate', dest='dropout_rate', default = 0.3,
                       help='Enter a Dropout Rate')
       

    parser.add_argument('--gpu', action="store_true", default=False,
                        help='Enter a Boolean Value, true or False. True will turn the GPU on.')
    
    args = parser.parse_args()
#     print(args)
    image_data_dir = args.image_data_dir
    PTmodel = args.pretrained_model
    checkpoint_path = args.checkpoint
    hidden_units = int(args.hidden_units)
    dropout = float(args.dropout_rate)
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    gpu = args.gpu
    
    
    #getting the directory and loading the data
    trainloader, validloader, testloader, train_data, valid_data, test_data = functions.loading_transforming_data(image_data_dir)
    
    #loading pretrained model from the argument
    model = getattr(models, PTmodel)(pretrained=True)
    
    #passing parameter and building classifier
    input_units = int(model.fc.in_features)
    model = functions.model_Classifier(model, input_units, hidden_units, dropout)
    
    #Defining loss function
    criterion = nn.NLLLoss()
    
    #defining optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=float(learning_rate))
    #training the model
    model, optimizer = functions.training_model(epochs,trainloader, validloader, gpu, model, criterion, optimizer)
    
    #testing the model
    functions.testing_model(model, testloader,gpu)
    
    #saving the model
    functions.saving_checkpoint(model, train_data, epochs, optimizer, checkpoint_path )
    

if __name__ == '__main__':
    main()
    