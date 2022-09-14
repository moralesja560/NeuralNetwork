from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
housing.data, housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(
X_train_full, y_train_full)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


model = keras.models.Sequential([
keras.layers.Dense(50, activation="relu", input_shape=X_train.shape[1:]),
#keras.layers.Dense(50, activation="relu"),
keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train_scaled, y_train, epochs=20,
validation_data=(X_valid_scaled, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[1:] # pretend these are new instances
print(X_new)
y_pred = model.predict(X_new)
print(y_pred)