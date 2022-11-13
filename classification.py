# Python packages
import numpy as np

from datetime import datetime

import utils
import train_model


# Load data
data = np.load('data_val.npz')

# Training set
X_train = data['X_train']
y_train = data['y_train']
# Validation set
X_val = data['X_val']
y_val = data['y_val']
# Test set
X_test = data['X_test']
y_test = data['y_test']

target_names = ['CD', 'HYP', 'MI', 'NORM', 'STTC']

# Get the model
# model_name = 'rajpurkar'
model_name = 'ribeiro'

timestamp = datetime.now().isoformat()
model_name = f'{model_name}-{timestamp}'

model = utils.get_model(X_train.shape, model_name)
# model.summary()

# Train the model
history = train_model.training(model, X_train, y_train, X_val, y_val, model_name)
print(history.params)
print(model_name)

# Evaluate the model
score = model.evaluate(X_test, y_test)
print(f"Custo de teste = {score[0]:.4f}")
print(f"Acurácia de teste = {100*score[1]:.2f}%")

# # Prediction of the model
# prediction = model.predict(X_test)
# # Convert the predictions to binary values
# prediction_bin = np.array(prediction)
# prediction_bin = (prediction > 0.5).astype('int')

# # Save results
# utils.get_metrics(y_test, prediction, prediction_bin, target_names)
# utils.plot_confusion_matrix(y_test, prediction_bin, model_name, target_names)
# utils.plot_results(history, name=model_name, metric='loss')
# utils.plot_results(history, name=model_name, metric='accuracy')
