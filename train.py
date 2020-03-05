import argparse # To create Parse argument
from utility_functions import * # to use the functions of the classifier
from workspace_utils import active_session # to active the session 
import sys

# Function to get input argument
def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line argumentsusing add_argument() from ArguementParser method
    parser.add_argument('data_dir', type = str, default = 'flowers', help = 'path to the folder flowers images')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help = 'directory to save checkpoints')
    parser.add_argument('--gpu',  action='store_true', help = 'set the train on gpu')
    parser.add_argument('--epochs',  type=int, default=20, help='number of epochs to train the model')  
    parser.add_argument('--learning_rate',  type=float, default=0.01, help = 'define the leaning for the optimizer')  
    parser.add_argument('--arch',type=str ,default = 'vgg', help = 'the CNN model architecture')
    parser.add_argument('--hidden_units', type=int, default=256, help= 'the number of hidden units')
 
    return parser.parse_args()


# The main function to lunch the train and save the model 
def main():
    # Get the arguments
    args = get_input_args()
    # Check if to use gpu 
    train_on_gpu = False
    if args.gpu:
        train_on_gpu = cuda.is_available() 
    # Define different directories of the data 
    data_dir = data_dirs(args.data_dir)
    # Create the data loaders
    image_datasets, dataloaders  = set_loaders(data_dir)
    # Load the pretrained model
    model = load_pretrained_model(args.arch)
    # Build the classifier
    model = build_classifier(model, args.arch,args.hidden_units, dataloaders)

    # training the model
    save_file_name = 'image_classifier.pt' # to save the sessions during the training phase
    train_model(model,
          train_on_gpu,
          args.learning_rate,
          dataloaders['train'],
          dataloaders['valid'],
          save_file_name,
          args.epochs,
          epochs_stop=3,
          print_every=2)   
    # Test the model
    test_model(model, dataloaders['test'], train_on_gpu)
    # Save the model
    save_checkpoint(model, args.epochs,args.learning_rate ,args.save_dir, image_datasets['train'])
    # complete.
    print('\n ## Training Successfully Completed! ## \n', flush = True)


if __name__ == "__main__":
    sys.stdout.flush()
    with active_session():
         main()