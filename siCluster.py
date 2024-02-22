import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from utils.siCluster_utils import *
from utils.parameters import *


def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    #Sets the random seed for reproducibility across different runs of the code
    path = args.pretrained_path
    #Assigns the path to the pretrained model from the command-line arguments
    model = models.resnet18(pretrained=False)
    #Initializes a ResNet-18 model with pre-trained weights set to False
    model = nn.DataParallel(model)
    #Wraps the model with nn.DataParallel to enable parallel training on multiple GPUs
    model.module.fc = nn.Linear(512, 10)
    #Replaces the fully connected layer (fc) of the model with a new linear layer with output size 10
    cudnn.benchmark = True
    #Sets the cuDNN benchmark to True for optimizing training performance
    model.load_state_dict(torch.load(path)['state_dict'], strict = False)
    #Loads the pretrained weights from the specified path into the model
    model.module.fc = nn.Sequential()
    #Replaces the fully connected layer with an empty sequential layer, effectively removing it
    model.cuda()
    #Moves the model to the GPU
    cudnn.benchmark = True
    #Sets the cuDNN benchmark to True for optimizing training performance
    
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #Defines a sequence of transformations to be applied to the input images for clustering.
#These transformations resize the image to 256x256, crop the center to 224x224, convert it to a PyTorch tensor, and normalize it.
    
    train_transform1 =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #Defines another sequence of transformations for training images.
#These transformations are similar to the cluster transformation but include additional random horizontal and vertical flips for data augmentation.
    train_transform2 =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #Defines a third sequence of transformations for the second set of training images, which is identical to train_transform1
    
    criterion = nn.CrossEntropyLoss().cuda()
    #Initializes two loss functions: cross-entropy loss (criterion) and a custom loss function (criterion2)
    criterion2 = AUGLoss().cuda()
#Moves both loss functions to the GPU
    if args.mode == 'rural':
        clusterset = GPSDataset('./meta_data/meta_rural.csv', './data/kr_data/', cluster_transform)
        trainset = GPSDataset('./meta_data/meta_rural.csv', './data/kr_data/', train_transform1, train_transform2)
    elif args.mode == 'city':
        clusterset = GPSDataset('./meta_data/meta_city.csv', './data/kr_data/', cluster_transform)
        trainset = GPSDataset('./meta_data/meta_city.csv', './data/kr_data/', train_transform1, train_transform2)
    else:
        raise ValueError
        #Depending on the mode specified in the arguments (args.mode), loads different datasets (clusterset and trainset) using GPSDataset
#Loads datasets with different transformations based on the mode
        
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=args.batch, shuffle=False, num_workers=1)
#creates data loaders (clusterloader and trainloader) using torch.utils.data.DataLoader to load batches of data from the datasets (clusterset and trainset). clusterloader is used for loading data during clustering and does not shuffle the data (shuffle=False).
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=1, drop_last = True)
#used for loading training data, shuffles the data (shuffle=True), and drops the last incomplete batch if any (drop_last=True)
    deepcluster = Kmeans(args.nmb_cluster)
#Initializes the clustering algorithm with the specified number of clusters (args.nmb_cluster) using the K-means algorithm.
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
#Initializes the Adam optimizer to optimize the parameters of the model with the specified learning rate (args.lr).
    
    
    for epoch in range(0, args.epochs): ##iterates over each epoch for the specified range and prints the current number using the print 
        print("Epoch : %d"% (epoch))
        features = compute_features(clusterloader, model, len(clusterset), args.batch)  #computes features for the data loaded by clusterloader using cluser_features and passing the model the length and the batch size
        clustering_loss, p_label = deepcluster.cluster(features) #uses deepcluser's object cluser to pass the computed features
        p_label = p_label.tolist() #this returns the clusetering loss nad the predicted cluster labels
        p_label = torch.tensor(p_label).cuda() #converts the predicted cluster labels to a list and then creates a tensor from it and moves it to the GPU

     
        model.train() #sets the training mode
        fc = nn.Linear(512, args.nmb_cluster) # initializes a fully connected layer (fc) w/ input size 512 and output size equal to number of clusters
        fc.weight.data.normal_(0, 0.01) #initializes the weights of the fully connected layer wiht a normal distribution and sets the biases to 0 in line below
        fc.bias.data.zero_()
        fc.cuda() # moves to GPU aka that is what .cuda does
        

        for batch_idx, (inputs1, inputs2, indexes) in enumerate(trainloader):
            #iterates over batches of data loaded by the trainloader, where each batch consists of inputs1, inputs2, and indexes
            inputs1, inputs2, indexes = inputs1.cuda(), inputs2.cuda(), indexes.cuda()  
            #moves those 3 to GPU using .cuda()         
            batch_size = inputs1.shape[0]
            #retrieves the batch size by accessing the size of the firest deimention of inputs1
            labels = p_label[indexes].cuda()
            #retrieves the cluster labels corresponding to teh current batch of data by indexing p_label wiht indexes and moves them to the GPU
            inputs = torch.cat([inputs1, inputs2])
            #concatenates inputs1 and inputs 2 alonf the first dimension to create a single input tensor
            outputs = model(inputs)
            #passes the concatendated inputs throguh the model to obtain the outputs
            outputs1 = outputs[:batch_size]
            outputs2 = outputs[batch_size:]
            #splits the outputs into two parts ooutputs1 for the first set of inputs (inputs1) and outputs2 for the second set (inputs2)
            outputs3 = fc(outputs1)
            #passes outputs1 throught the fully connected layer fc, to obtain output3
            ce_loss = criterion(outputs3, labels)
            #computs the cross-entropy loss (ce_loss) between outputs3 and labels
            aug_loss = criterion2(outputs1, outputs2) / batch_size
            #computes the augmentation loss (aug_loss) using the cirterion2 function and divides it by the batch size
            loss = ce_loss + aug_loss
            #computes teh total loss (loss) as the sum of teh cross-entropy loss and the augmentation loss
            optimizer.zero_grad()
            #initializes the gradients to zero usign optimizer.zero_grad() to clear any previously stored gradients
            loss.backward()
            #computes the gradients of the lsos w/ respect to the model parameters using backpropogation (loss.backward())
            optimizer.step()
            #updates the model parameters using the optimizer

            if batch_idx % 20 == 0:
                print("[BATCH_IDX : ", batch_idx, "LOSS : ",loss.item(), "CE_LOSS : ",ce_loss.item(),"AUG_LOSS : ",aug_loss.item(),"]" )
                #prints the loss information every 20 batched if the batch index (batch_idx) is a multiple of 20
    torch.save(model.state_dict(), './checkpoint/ckpt_cluster_{}.t7'.format(args.mode))
    #saves the state of the model (model.state_dict()) to a file named ckpt_cluster_{}.t7 where {} is replaced by the mode specified in args
                                                       
    
if __name__ == "__main__":
    args = siCluster_parser()
    main(args)    
    #parses the command-line arguments using siCluster_parser() and calls the main fucntion with the parsed arguments
    
