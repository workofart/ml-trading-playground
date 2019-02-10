import numpy as np, pandas as pd
from utils import initialize_parameters, initialize_parameters_deep, linear_forward, linear_activation_forward, L_model_forward, compute_cost, linear_backward, linear_activation_backward, L_model_backward, update_parameters, predict, L_layer_model, normalize, read_data, generate_datasets
    

data = read_data('crypto-test-data-82hrs.csv')
data = data[data['ticker'] == 'ETHBTC']
data['timestamp'] =pd.to_datetime(data.timestamp)
data = data[['high', 'low', 'price', 'volume', 'timestamp']].sort_values(by='timestamp')
data = data.set_index('timestamp')

X_train, X_test, Y_train, Y_test = generate_datasets(data)
X_train = normalize(X_train)
X_test = normalize(X_test)

print('Training set shape: ' + str(X_train.shape))
print('Testing set shape: ' + str(X_test.shape))

# 1. Initialize parameters / Define hyperparameters
# 2. Loop for num_iterations:
#     a. Forward propagation
#     b. Compute cost function
#     c. Backward propagation
#     d. Update parameters (using parameters, and grads from backprop) 
# 4. Use trained parameters to predict labels

layers_dims = [X_train.shape[0], 16, 6, Y_train.shape[0]] #  4-layer model

# Current Cost after iteration 5900: 0.000760 (without normalization)
# Cost after iteration 5900: 0.291479 (with normalization)
parameters = L_layer_model(X_train, Y_train, layers_dims, learning_rate=0.03, print_cost=True)

pred_train = predict(X_train, Y_train, parameters, 'train')
pred_test = predict(X_test, Y_test, parameters, 'test')