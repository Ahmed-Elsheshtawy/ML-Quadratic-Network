import random
from math import e

# I will try to have as much inputs as possible to train and test a model that learns the function f(x) = x^2
training_inputs = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
training_outputs = [100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Additional testing inputs and outputs to evaluate the model's performance on unseen data
testing_inputs = [-15, -12, -11, -0.5, 0.5, 11, 12, 15]
testing_outputs = [225, 144, 121, 0.25, 0.25, 121, 144, 225]

class Model:
    def __init__(self):
        # Initialize with small random weights for better learning
        self.w3 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)
        self.w1 = random.uniform(-1, 1)
        self.b3 = random.uniform(-1, 1)
        self.b2 = random.uniform(-1, 1)
        self.b1 = random.uniform(-1, 1)
        self.dL_dw3 = 0
        self.dL_dw2 = 0
        self.dL_dw1 = 0
        self.dL_db3 = 0
        self.dL_db2 = 0
        self.dL_db1 = 0
        self.n1_input = 0
        self.n1_output = 0
        self.n2_input = 0
        self.n2_output = 0
    
    def sigmoid(self, x):
        # Numerically stable sigmoid to avoid overflow
        if x >= 0:
            return 1 / (1 + e ** (-x))
        else:
            exp_x = e ** x
            return exp_x / (1 + exp_x)

    def feedforward(self, x):
        # Feed forward through the network for the given input x
        self.n1_input = self.w1 * x + self.b1
        # Clip input to sigmoid to prevent overflow
        self.n1_input = max(min(self.n1_input, 500), -500)
        self.n1_output = self.sigmoid(self.n1_input)
        self.n2_input = self.w2 * self.n1_output + self.b2
        self.n2_output = self.n2_input  # Linear activation
        output = self.w3 * self.n2_output + self.b3
        # Clip output to prevent overflow
        return max(min(output, 1e6), -1e6)
    
    def calculate_loss(self, predicted, correct):
        return (predicted - correct) ** 2
    
    def calculate_gradients(self, x, predicted, correct):
        dL_dy = 2 * (predicted - correct)
        # Clip gradient to prevent explosion
        dL_dy = max(min(dL_dy, 1e6), -1e6)
        
        self.dL_dw3 = max(min(dL_dy * self.n2_output, 1e6), -1e6)
        self.dL_db3 = dL_dy

        self.dL_dw2 = max(min(dL_dy * self.w3 * self.n1_output, 1e6), -1e6)
        self.dL_db2 = max(min(dL_dy * self.w3, 1e6), -1e6)

        # Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
        # Using stored self.n1_output which is sigmoid(self.n1_input)
        sigmoid_derivative = self.n1_output * (1 - self.n1_output)
        
        self.dL_dw1 = max(min(dL_dy * self.w3 * self.w2 * x * sigmoid_derivative, 1e6), -1e6)
        self.dL_db1 = max(min(dL_dy * self.w3 * self.w2 * sigmoid_derivative, 1e6), -1e6)
        # Now the gradients are stored in self.dL_dw3, self.dL_db3, self.dL_dw2, self.dL_db2, self.dL_dw1, self.dL_db1
    
    def update_parameters(self, learning_rate):
        self.w3 -= learning_rate * self.dL_dw3
        self.b3 -= learning_rate * self.dL_db3
        self.w2 -= learning_rate * self.dL_dw2
        self.b2 -= learning_rate * self.dL_db2
        self.w1 -= learning_rate * self.dL_dw1
        self.b1 -= learning_rate * self.dL_db1
    
    def train(self, inputs, correct_outputs, epochs, learning_rate):
        for epoch in range(epochs):
            total_loss = 0
            for x, correct_y in zip(inputs, correct_outputs):
                predicted_y = self.feedforward(x)
                total_loss += self.calculate_loss(predicted_y, correct_y)
                self.calculate_gradients(x, predicted_y, correct_y)
                self.update_parameters(learning_rate)
            avg_loss = total_loss / len(inputs)
            print("Epoch", epoch + 1, "Loss:", avg_loss)
    
    def test(self, inputs, correct_outputs):
        total_loss = 0
        predicted_outputs = []
        for x, correct_y in zip(inputs, correct_outputs):
            predicted_y = self.feedforward(x)
            predicted_outputs.append(predicted_y)
            total_loss += self.calculate_loss(predicted_y, correct_y)
        loss = total_loss / len(inputs)
        print("Test Loss:", loss)
        print("Correct Outputs:", correct_outputs)
        print("Predicted Outputs:", predicted_outputs)
        return predicted_outputs

model = Model()
model.train(training_inputs, training_outputs, epochs=1000, learning_rate=0.001)
model.test(testing_inputs, testing_outputs)