this file is more explanations on the detail of the code for me to better undestand what is going on. 

## Starting with extract_cluster.py:

### Line 20: 
**convent** is the "convolutional neural network model" defined in pytorch

**torch.nn.DataParallel** the next part is a module that allows you to parallelize the operations of a modle across multiple GPUs. It works by splitting the input data into chunks and then processing each one on a seperate GPU. After processing, it gathers the result and combines them. 

Here we passs convent into it, which means we are "wrapping" the convent with DataParallel; 

**wrapping** is extending the existing behavior without altering it. 

this is basically and efficiency statement, as we are tryign to split the computation between GPUs rather than doing it all at once. 

### Line 29:
**transforms.ToTensor()** converts the dimensions of the image from Height x Width x Channels (H x W x C) to Channels x Height x Width (C x H x W), which is the standard format for PyTorch tensors representing images.

the range also changes from [0,255] to [0.0, 1.0]

This transformation is often used in combination with other transformations when constructing a torchvision.transforms.Compose pipeline for preprocessing image data before feeding it into a neural network model.

### Line 30: 
** transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) **

Mean: This parameter is a list or tuple containing the mean values for each channel of the image. For RGB images, there are typically three channels corresponding to red, green, and blue. In this case, [0.485, 0.456, 0.406] represents the mean values for the red, green, and blue channels, respectively.

Standard deviation: Similarly, this parameter is a list or tuple containing the standard deviation values for each channel of the image. [0.229, 0.224, 0.225] represents the standard deviation values for the red, green, and blue channels, respectively.

When applied to an image, the transforms.Normalize transformation performs the following operations for each channel:

Subtracts the mean value.
Divides by the standard deviation value.
