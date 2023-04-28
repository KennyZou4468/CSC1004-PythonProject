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

5. def plot:
   easy to understand. We input epoches and coresponding value(accuracy,loss ...)...and plot them using matplot.

6.
