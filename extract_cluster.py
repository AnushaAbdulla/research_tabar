import os
import torch
import torch.nn as nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.siCluster_utils import * #custom utility from other files
from utils.parameters import * #^^
import glob
import shutil
import copy
import csv


def extract_city_cluster(args):
    convnet = models.resnet18(pretrained=True) #ResNet18 - a pretrained model on ImageNet dataset, is being used to extract features from images 
    convnet = torch.nn.DataParallel(convnet) #wraps the model with 'DataParallel" to parallize operations across multiple GPUs

    ckpt = torch.load('./checkpoint/{}'.format(args.city_model)) #loads the pre-trained weights of the ResNet18 model from a checkpoint file specific to the city model providede in the arguments. 
    convnet.load_state_dict(ckpt, strict = False)#Loads the model's state dictionary with the loaded weights, allowing the model to use pre-trained weights for feature extraction
    convnet.module.fc = nn.Sequential()#Removes the fully connected layer (fc) from the ResNet18 model. This layer is typically used for classification, but since this model is being used for feature extraction only, the classification layer is removed
    convnet.cuda() #Moves the model to the GPU for faster computation
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    #Defines a series of image transformations to be applied to each image in the dataset. These transformations include resizing, center cropping, converting to a tensor, and normalization
    
    clusterset = GPSDataset('./meta_data/meta_city.csv', './data/kr_data/', cluster_transform) #Initializes a dataset (clusterset) using the GPSDataset class with metadata CSV file (meta_city.csv), image directory (kr_data), and the defined transformation (cluster_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=256, shuffle=False, num_workers=1) #Creates a DataLoader (clusterloader) to load data from the clusterset dataset. It loads data in batches of size 256, does not shuffle the data, and uses a single worker for data loading
    
    deepcluster = Kmeans(args.city_cnum) #Initializes a Kmeans clustering object (deepcluster) with the number of clusters specified in the arguments for the city
    features = compute_features(clusterloader, convnet, len(clusterset), 256) #Computes features from the images in the dataset using the defined model (convnet) and DataLoader (clusterloader). The compute_features function likely extracts features from intermediate layers of the model
    clustering_loss, p_label = deepcluster.cluster(features) #Performs clustering on the extracted features using the initialized deepcluster object. It returns the clustering loss and predicted labels for each image.
    labels = p_label.tolist() #Converts the predicted labels from PyTorch tensors to a list for easier manipulation
    f = open('./meta_data/meta_city.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    #Opens the metadata CSV file for the city dataset and reads the image filenames into a list (images)
    images.pop(0) #Removes the header (first element) from the images list since it contains column names
    city_cluster = []
    for i in range(0, len(images)):
        city_cluster.append([images[i], labels[i]]) #Creates a list of tuples (city_cluster) where each tuple contains the image filename and its corresponding cluster label
        #A tuple in Python is an ordered collection of elements, similar to a list. 
        
    return city_cluster #Returns the list of image filenames and their cluster labels for the city dataset

def extract_rural_cluster(args):
    convnet = models.resnet18(pretrained=True) #Initializes a ResNet18 model pretrained on the ImageNet dataset. This model will be used for feature extraction from images
    convnet = torch.nn.DataParallel(convnet) #Wraps the model with DataParallel to parallelize operations across multiple GPUs if available    
    ckpt = torch.load('./checkpoint/{}'.format(args.rural_model)) #Loads the pre-trained weights of the ResNet18 model from a checkpoint file specific to the rural model provided in the arguments
    convnet.load_state_dict(ckpt, strict = False) #Loads the model's state dictionary with the loaded weights, allowing the model to use the pre-trained weights for feature extraction
    convnet.module.fc = nn.Sequential() #Removes the fully connected layer (fc) from the ResNet18 model. This layer is typically used for classification, but since this model is being used for feature extraction only, the classification layer is removed
    convnet.cuda() #Moves the model to the GPU for faster computation
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    
    #Defines a series of image transformations to be applied to each image in the dataset. These transformations include resizing, center cropping, converting to a tensor, and normalization
    clusterset = GPSDataset('./meta_data/meta_rural.csv', './data/kr_data/', cluster_transform) #Initializes a dataset (clusterset) using the GPSDataset class with metadata CSV file (meta_rural.csv), image directory (kr_data), and the defined transformation (cluster_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=256, shuffle=False, num_workers=1) #Creates a DataLoader (clusterloader) to load data from the clusterset dataset. It loads data in batches of size 256, does not shuffle the data, and uses a single worker for data loading
    
    deepcluster = Kmeans(args.rural_cnum) #Initializes a Kmeans clustering object (deepcluster) with the number of clusters specified in the arguments for the rural areas
    features = compute_features(clusterloader, convnet, len(clusterset), 256) #Computes features from the images in the dataset using the defined model (convnet) and DataLoader (clusterloader). The compute_features function likely extracts features from intermediate layers of the model
    clustering_loss, p_label = deepcluster.cluster(features) #Performs clustering on the extracted features using the initialized deepcluster object. It returns the clustering loss and predicted labels for each image
    labels = p_label.tolist() #Converts the predicted labels from PyTorch tensors to a list for easier manipulation
    f = open('./meta_data/meta_rural.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    #Opens the metadata CSV file for the rural dataset and reads the image filenames into a list (images)
    images.pop(0) #Removes the header (first element) from the images list since it contains column names
    rural_cluster = []
    for i in range(0, len(images)):
        rural_cluster.append([images[i], labels[i] + args.city_cnum])
    #Creates a list of tuples (rural_cluster) where each tuple contains the image filename and its corresponding cluster label. The rural cluster labels are incremented by the number of city clusters to ensure uniqueness across all clusters
    return rural_cluster
# Returns the list of image filenames and their cluster labels for the rural dataset

def extract_nature_cluster(args): #Defines a function named extract_nature_cluster which takes args as an argument
    f = open('./meta_data/meta_nature.csv', 'r', encoding='utf-8')
    #Opens the metadata CSV file specific to the nature dataset for reading
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
        #Reads each line from the CSV file using a CSV reader (rdr) and appends the first element of each line (which is assumed to be the image filename) to the images list
    f.close()
    #Closes the file after reading all the lines
    images.pop(0)   
    #Removes the first element from the images list, assuming it's the header of the CSV file
    nature_cluster = []
    cnum = args.city_cnum + args.city_cnum #CONFUSION
    #Initializes an empty list nature_cluster to store the nature cluster information.
#Calculates the total number of city clusters by doubling the number of city clusters (args.city_cnum) and assigns it to the variable cnum
    for i in range(0, len(images)):
        nature_cluster.append([images[i], cnum])
        #Iterates over the images list and appends each image filename along with the calculated cnum to the nature_cluster list
            
    return nature_cluster
#Returns the list of image filenames and their corresponding cluster label (which is the calculated cnum) for the nature dataset


def main(args):
    # make cluster directory
    city_cluster = extract_city_cluster(args)
    rural_cluster = extract_rural_cluster(args)
    nature_cluster = extract_nature_cluster(args)
    #Calls three functions (extract_city_cluster, extract_rural_cluster, and extract_nature_cluster) to extract clusters for city, rural, and nature images based on the provided arguments (args)
    total_cluster = city_cluster + rural_cluster + nature_cluster
    #Concatenates the clusters obtained from different regions into a single list called total_cluster
    cnum = args.city_cnum + args.rural_cnum #CONFUSION
    #Calculates the total number of clusters (cnum) by summing the number of city clusters and rural clusters
    cluster_dir = './data/{}/'.format(args.cluster_dir)
    #Defines the directory path where the clusters will be stored based on the provided argument cluster_dir
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
        for i in range(0, cnum + 1):
            os.makedirs(cluster_dir + str(i))
    else:
        raise ValueError
    #Checks if the cluster directory exists. If not, it creates the directory along with subdirectories for each cluster. If the directory already exists, it raises a ValueError
    for img_info in total_cluster:
        cur_dir = './data/kr_data/' + img_info[0]
        new_dir = cluster_dir + str(img_info[1])
        shutil.copy(cur_dir, new_dir)
        #Copies the images from their original location (./data/kr_data/) to their respective cluster directories based on the cluster information (img_info) obtained earlier
    
    # make cluster census histogram for census mode
    data_dir = './data/kr_data/'
    cluster_list = []
    for i in range(0, cnum):
        cluster_list.append(os.listdir(cluster_dir + str(i)))
    cluster_score = []    
    for i in range(0, cnum):
        cluster_score.append(0)
    #Initializes variables for creating a cluster census histogram. It creates a list (cluster_list) containing the filenames in each cluster and initializes a list (cluster_score) to keep track of the count of images in each cluster
    histogram_dir = cluster_dir + args.histogram
    f = open(histogram_dir, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
#Defines the file path for the histogram file based on the provided argument histogram_dir and opens a CSV file in write mode for writing the histogram data
    for i in range(1, 231):
        r_list = os.listdir(data_dir + str(i))
        r_score = copy.deepcopy(cluster_score)
        for region in r_list:
            for i in range(0, len(cluster_list)):
                if region in cluster_list[i]:
                    r_score[i] += 1
                    break
        wr.writerow(r_score)
    f.close()
    #Iterates over a range of values representing different regions. For each region, it counts the number of images in each cluster and writes the histogram data to the CSV file
    # make metadata for cluster && total dataset for eval
    file_list = glob.glob("./{}/*/*.png".format(args.cluster_dir))
    grid_dir = cluster_dir + args.grid
    f = open(grid_dir, 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['y_x', 'cluster_id'])
    #Defines the file path for the metadata file based on the provided argument grid_dir and opens a CSV file in write mode for writing metadata
    for file in file_list:
        file_split = file.split("/")
        folder_name = file_split[2]
        file_name = file_split[-1].split(".")[0]
        wr.writerow([file_name, folder_name])
    f.close()
    #Iterates over the list of image files in the cluster directory, extracts the folder name and file name, and writes them to the metadata CSV file
        
    if not os.path.exists('./data/cluster_kr_unified'):
        os.makedirs('./data/cluster_kr_unified')
    for i in range(cnum + 1):
        file_dir = cluster_dir + '{}/*.png'
        file_list = glob.glob(file_dir.format(i))    
        for cur_dir in file_list:
            shutil.copy(cur_dir, './data/cluster_kr_unified')
    #Checks if the directory for the unified cluster dataset exists. If not, it creates the directory. Then, it iterates over each cluster, copies all images in each cluster to the unified cluster directory.
if __name__ == "__main__":
    args = extract_cluster_parser()
    main(args)    
    #Checks if the script is being run directly. If so, it extracts arguments using a parser function (extract_cluster_parser()) and calls the main function with these arguments. This ensures that the main function is executed when the script is run directly
