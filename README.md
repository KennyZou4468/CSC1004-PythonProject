# CSC1004-PythonProject


This is a tutorial about how to implement the function in this project.

1.Class Net:
  This is the structure of the neural network.
  
2. def accuracy_fn:
  this is a function defined by me to calculate the accuracy.
  
3.def train:
  Firstly, we turn the mode to train and initialize running accuracy and running loss to be 0. Then
  we take out data and target from train_loader and put into device. Then use "F.nll_loss"to calculate
  loss, make optimizer gradient become 0, backward loss and step optimizer.
  Next, we add up loss item and use previous function to calculate accuracy. Then we output result. 
  To get rate, we finally devide each other by the length of train_loader. We also pipe our result to txt file.
  
4. def test:
   Firstly we change to evalation mode and initialize. Then we remove gradient by calling "with torch.inference_mode" and almost repeat same steps in train function.
   
Remark: output are predicted result from the machine, and we compare them to known target.

5. def plot& def plot_mean:
   easy to understand. We input epoches and coresponding value(accuracy,loss ...)...and plot them using matplot.

6.def Run:
  Like the main function, containing almost the key function inside.when constructing train_loader and test_loader, we use the result in "seed_worker" and put them into the dataset. For every epoch,we call train function and test function and store the values into list(Array).After 15 epoches, we call plot function to plot them. When we want to plot mean values, we create other 4 lists in main function and use Manager to manage them and get their values through training,after run function we append training acc/loss, testing acc/ loss to corresponding lists.Finaly we get 4 list with size 3 *15. We use mutiprocessing by using 3 process and start and join them. We initialize other 4 mean value list with sizes we known. Then we add mean value in each epoch for three run to them and get 15 mean values. After that we will get final mean values and plot them.
 
 This is how the project works
  You can find the plots in the floder "plot"
  
  
  
  
  
  
  
