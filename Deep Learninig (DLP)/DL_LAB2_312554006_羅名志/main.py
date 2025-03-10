"""
    National Yang Ming chiao Tung University
    Deep Learning Lab2, Butterfly_Moth_Classification
    Student ID: 312554006
    Student Name: Ming Chih, Lo
    Department: Data Science and Engineering, College of Computer Science
    Contact mail: max230620089@gmail.com

    Goal: implement Vgg19 and ResNet50 and compare their results.

    Input data: as defined in dataloader.py
    Output: train/test result and model weights.

    Ref.
    1. Slide and Spec provided on E3 platform
    2. Architecture of ResNet50: https://ithelp.ithome.com.tw/articles/10264843
    3. Sample code of ResNet50: https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py (the make_layer function is referenced from the github repo)
    4. Architecture of VGG19: https://ithelp.ithome.com.tw/articles/10192162
    5. Sample code of VGG19: https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-vgg19.ipynb
    6. For dataloader and data augmentation: https://pytorch.org/vision/stable/transforms.html

"""


from dataloader import BufferflyMothLoader
import torch
import torch.nn as nn
import torch.optim as optim
import ResNet50
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import os 
from ResNet50 import ResNet50
from VGG19 import VGG19
from torch.utils.tensorboard import SummaryWriter



def evaluate(model, data_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct_prediction = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_prediction += (predicted == labels).sum().item()
            
    avg_loss = total_loss/len(data_loader.dataset)
    accuracy = 100*correct_prediction/len(data_loader.dataset)
    return avg_loss, accuracy


def test(model, test_loader, criterion):
    return evaluate(model, test_loader, criterion)

def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, path = 'model_weights/ResorVGG.pth'):
    print("start training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # initialize recording variables
    writer = SummaryWriter(filename_suffix = path.split('/')[-1].split('.')[0])
    best_valid_loss = float('inf')
    best_model_weights = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_prediction = 0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave = False) as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameters gradients
                optimizer.zero_grad()

                # foward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # backward
                loss.backward()
                optimizer.step()

                # update the progress bar with current loss
                running_loss += loss.item()*inputs.size(0)
                t.set_postfix(loss=loss.item())
        
            # calculate loass and accuracy
                _, predicted = torch.max(outputs, 1)
            
                correct_prediction += (predicted == labels).sum().item()      
        
        # recore training acc and loss
        train_loss, train_acc = running_loss/len(train_loader.dataset), 100*correct_prediction/len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}%")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # Evaluate on the validation set
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {valid_loss:.4f}, Validation accuracy: {valid_acc:.4f}%")
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_weights = model.state_dict()
    
    torch.save(best_model_weights, path)
    writer.close()



if __name__ == "__main__":
    
    """
    Useage:
        python main.py --learning_rate 0.0001 --num_epochs 60 --batch_size 64 --model ResNet50 --demo 0
    Params: 
        learning_rate: Learning rate for training, default = 0.0001
        num_epochs: Number of epochs for training, default = 60
        batch_size: Batch_size for training, default = 64
        model: Model to use, ResNet50 or VGG19, default = ResNet50
        demo: For demo the best model, default = 1(for demo)
    Result:
        model weights(for loading in demo): saved in ./model_weights
        loss/acc: saved in ./test_result
        visualization: using tensorboard, saved in ./runs   
    """

    # Check if CUDA is available
    if torch.cuda.is_available():
    # Get the currently selected CUDA device
        device = torch.cuda.current_device()
        print(f"Current CUDA device: {torch.cuda.get_device_name(device)}")
        
    # Load parameters
    parser = argparse.ArgumentParser(description = "Script for demo")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=60, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--model", type=str, default='VGG19', help="Model for training")
    parser.add_argument("--demo", type=int, default=1, help='0 for train a new model, 1 for load parameters')
    args = parser.parse_args()
    print(args)
    
    # Define hyperparameters
    lr = args.learning_rate
    epochs = args.num_epochs
    batch = args.batch_size
    if args.model == 'ResNet50':
        model = ResNet50([3,4,6,3], 3, 100) 
    else:
        model = VGG19(3, 100)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Read data
    # file location
    root = os.getcwd()
    path = root + f'/model_weights/best_{args.model}.pth'
    # dataloader for train, validation, test
    train_dataset = BufferflyMothLoader(root=root, mode ='train')
    valid_dataset = BufferflyMothLoader(root=root, mode ='valid')
    test_dataset = BufferflyMothLoader(root=root, mode ='test')

    train_loader = DataLoader(train_dataset, batch_size=batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)

    # train
    if args.demo == 0:
        train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=epochs, path=path)

    # test
    model.load_state_dict(torch.load(path))
    train_loss, train_acc = test(model, train_loader, criterion)
    test_loss, test_acc = test(model, test_loader, criterion)

    # Result
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    test_path = root + f'/test_result/best_{args.model}.txt'
    with open(test_path, 'w') as f:
        # Write the testing result to the file
        f.write(f"model: {args.model}\n")
        f.write(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}%\n")
