# Neural Network made by Avi Mathur.

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
Gradient descent:
15. W(1) = W(1) - learning_rate * dW(1).
16. b(1) = b(1) - learning_rate * db(1).
17. W(2) = W(2) - learning_rate * dW(2).
18. b(2) = b(2) - learning_rate * db(2). 
repeat steps 5-18 for each epoch.

Additional notes:
1. Each column of A(0) is one image:

    So multiplying W(1) × A(0) gives:

    (10 × 784) × (784 × m) → (10 × m)

    So you get 10 outputs (1 per hidden neuron) for each of the m inputs.
    It adds b(1) to each column of Z(1) (i.e., each training example).
    This gives the correct shape: still (10 × m)

2. ReLU is x for x>=0, else 0.

3.Softmax converts raw scores (logits) into probabilities that are non-negative & sum to 1.

4. In backpropagation, the first step is to calculate the gradient of the loss function wrt Z(2), which is the output layer's pre-activation values.
   This is done by subtracting the true labels Y from the predicted probabilities A(2).
   This matrix tells you how much each score in Z(2) contributed to the error in the loss.

5. One-hot encoding is a way to represent a class label (like “5”) as a 10-length vector, where only index 5 is 1 (hot) and all others are 0.

6. Then we are calculating how much each weight in W2 contributed to the error in Z2, by multiplying dZ(2) with the activation that led to it A1.T.
   We average over m to make it a mean gradient across the batch.
   W2 -= learning_rate * dW2

7. Then we are computing the gradient of the loss dz(2) w.r.t. the bias term in the final layer. 
    axis=1 would sum each row (i.e. over the m examples)
    keepdims=True keeps the result as a column vector → (10, 1)

8. dA(1) gives us how much the output error affects each hidden unit’s activation.

9. dZ(1) is element-wise multiplication of the backpropagated gradient with the ReLU derivative.
   ReLU's derivative is 1 for positive inputs and 0 for negative inputs, so when Z1<=0, we zero out the gradient.
   
10. dW(1) calculating how each input pixel X.T influenced the loss via each hidden unit dZ(1).
    Then db(1) same as db(2).

11. Learning rate is a small positive number (like 0.01) that controls how big the step is in each update of the model parameters. Can't be too small or too large(slow, overshoot respectively).
    To minimize the loss, and gradient descent moves in the direction that reduces it most quickly. After many such steps, the weights converge to values that minimize the loss.

UPDATE: Instead of using VSCode, I am using Kraggle Notebook for easy input of Mnist dataset. 
I have linked the notebook in the repository.
