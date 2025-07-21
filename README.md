# NeuralNetwork

A basic neural network implemented from scratch using NumPy.  
This project is for learning core deep learning concepts like forward propagation, backpropagation, and training loops.
Without using libraries like Tensorflow & Pytorch to better understand the subject.

Math:
1. Using the MNIST dataset, the neural network is trained to recognize handwritten digits(0-9), i.e 10 classes.
2. Each image is a 28x28 pixel grayscale image(each pixel is from 0-255, black to white), made into a matrix of m columns with 784 rows each.
3. Input layer with 784 nodes, first hidden layer with 10 nodes, second output layer also with 10 nodes.
4. A(0) = X, where X is the input matrix(784xm).
5. Z(1)(10xm) = W(1)(10x784) * A(0)(784xm) + b(1)(10x1).
6. A(1) = ReLU(Z(1)).
