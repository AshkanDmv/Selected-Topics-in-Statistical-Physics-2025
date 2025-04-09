"""At first, I import numpy library to work matrixes and other mathematical tools for this problem. """
import numpy as np
"""Now I'm going to define a function for ridge regression with gradient descent algorithm. This function will take five inputs:
1. A: This parameter takes the matrix of coefficients.
2. B: This parameter takes the vector that contains results.
3. alfa: This parameter takes the regularization strength for our algorithm.
4. stepsize: This parameter takes the step size (learning rate) that is needed for the algorithm.
5. maxiteration: This parameter takes the maximum number of iteration (updates) that is allowed for the algorithm.
Next, I'm going to define a function that is for the use of ridge regression for gradient descent algorithm."""
def ridreg(A, B, alfa, stepsize, maxiteration):
    num_samples, num_features= A.shape
    w = np.zeros(num_features)
    for i in [0, maxiteration]:
      B_prediction = A @ w
      error = (B_prediction) - B
      """Now I'm going to define a variable that represents the gradient of the cost function with ridge penalty and 
      then update the weights (w)."""
      grad = ((2 / (num_samples)) * (A.T @ error)) + (2 * alfa * w)
      w = w - (stepsize * grad)
    return w

"""Now I'm going to define a function that will randomly generate the data that is needed for the ridge regression."""
def datagen(num_samples, num_features, w0, noise):
    A = np.random.randn(num_samples, num_features)
    A1 = (10**(-4)) / np.linalg.det(A)
    A = A * A1
    B = (A @ w0) + (np.random.normal(0, noise, size = num_samples))
    return A, B

"""The next function is defined to input needed parameters by the user and then, we will finally see the results of the code"""
def main():
    num_samples = int(input("Hi! At first, please enter the number of samples: "))
    num_features = int(input("Now in this step, enter the number of features: "))
    w0 = np.array(list(map(float, input(f"In this step, enter {num_features} real weights: ").strip().split())))
    noise = float(input("In this step, enter the noise: "))
    alfa = float(input("In this step, enter regularization strength: "))
    stepsize = float(input("In this step, enter the step size: "))
    maxiteration = int(input("In this step, enter the maximum number of iteration: "))
    A, B = datagen(num_samples, num_features, w0, noise)
    wi = ridreg(A, B, alfa, stepsize, maxiteration)
    print("The final weights are: ", wi)
    print("Real weights :", w0)
if __name__ == "__main__":
    main()
