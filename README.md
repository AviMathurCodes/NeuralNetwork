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
7. Z(2)(10xm) = W(2)(10x10) * A(1)(10xm) + b(2)(10x1).
8. A(2) = Softmax(Z(2)).
backpropagation:
9. dZ(2) = A(2) - Y, where Y is the hot-encoded output matrix.
10. dW(2) = dZ(2) * A(1).T / m.
11. db(2) = np.sum(dZ(2), axis=1, keepdims=True) / m.
12. dZ(1) = dZ(2) * W(2).T.*g'(Z(1)), where g' is the derivative of ReLU.
13. dW(1) = dZ(1) * X.T / m.
14. db(1) = np.sum(dZ(1), axis=1, keepdims=True) / m.
learning rate:
15. W(1) = W(1) - learning_rate * dW(1).
16. b(1) = b(1) - learning_rate * db(1).
17. W(2) = W(2) - learning_rate * dW(2).
18. b(2) = b(2) - learning_rate * db(2). 
repeat steps 5-18 for each epoch.