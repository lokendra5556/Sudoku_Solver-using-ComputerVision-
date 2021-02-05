# Sudoku_Solver
  this was a fun project , basically what I did was to make a program that can read any sudoku image, extract numbers and their positions from it and try to predict what can be in the blank space
# How it works
  first of all we need to capture data from the image, for that first we  finds contours, sort them and based on the contours crop 81 parts from image, and save all of them into a folder named cropped then apply a machine learning model to predict number using each cropped part and this way we get an array of number then we pass it through a function that finds solution of that sudoku using dynamic programming
# Training
  to train a model that can predict the number i used MNIST_DATASET it is a tabular dataset having binary value of each 28*28 pixel,but in our case 0 is just blank space thats why i changed all pixel values to 0 where label is 0,i have used Support Vector Classifier to predict the numbers
### It is a POC project , and needs further improvement  
