this file is more explanations on the detail of the code for me to better undestand what is going on. 

## Starting with extract_cluster.py:

### Line 20: 
**convent** is the "convolutional neural network model" defined in pytorch

**torch.nn.DataParallel** the next part is a module that allows you to parallelize the operations of a modle across multiple GPUs. It works by splitting the input data into chunks and then processing each one on a seperate GPU. After processing, it gathers the result and combines them. 

Here we passs convent into it, which means we are "wrapping" the convent with DataParallel; 

**wrapping** is extending the existing behavior without altering it. 

this is basically and efficiency statement, as we are tryign to split the computation between GPUs rather than doing it all at once. 
