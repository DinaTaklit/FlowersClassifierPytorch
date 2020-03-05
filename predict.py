import argparse
from utility_functions import * # to use the functions of the classifier

# Function to get input argument
def get_input_args():    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    # Create command line argumentsusing add_argument() from ArguementParser method
    parser.add_argument('image', type = str, help = 'image path')
    parser.add_argument('checkpoint', type=str, help = 'saved model path')
    parser.add_argument('--top_k', type=int, default=5, help = 'top k most probable classes')
    parser.add_argument('--category_names', type=str, default=' ', help = 'json file use a mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help = 'enable training on gpu')
    return parser.parse_args()


# the main function to launch the whole process
def main():
    #Get input arguments
    args = get_input_args()
    train_on_gpu = False
    if args.gpu:
        train_on_gpu = cuda.is_available() 
    # Load the trained model
    model = load_checkpoint(args.checkpoint)
    # Process the image
    image = process_image(args.image)
    # Class prediction
    top_probs, top_classes, top_classes_index = predict(image, model, train_on_gpu, args.top_k)
  
    # printing the results
    if args.category_names != ' ':   
        indexes = []
        names = [] 
        # loading the json file of the category names
        print('topclasindex=>{}'.format(top_classes_index))
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)  
        
        for result in range(len(top_probs)):
            print('Rank=> {:<2}| Class=> {:<4}| Proba=> {:.4f}\n'.format(result + 1, top_classes[result], top_probs[result]))          
    else:
        for result in range(len(top_probs)):
            print('Rank=> {:<2}| Class=> {:<4}| Proba=> {:.4f}\n'.format(result + 1, top_classes[result], top_probs[result]))


if __name__ == "__main__":
    main()
