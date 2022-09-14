from keras.datasets import fashion_mnist
import tensorflow as tf


# import the data
(X_train_full,y_train_full),(X_test,y_test) = fashion_mnist.load_data()

#split the data
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model1 = tf.keras.Sequential([
	#tf.keras.layers.Flatten(input_shape=[28,28]),
	#tf.keras.layers.Dense(300, activation="relu"),
	#tf.keras.layers.Dense(100, activation="relu"),
	#tf.keras.layers.Dense(10, activation="softmax")])
	tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(1,1),padding= 'valid', activation = "relu", input_shape =(28,28,1)),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Conv2D(32,3,activation="relu"),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Conv2D(64,3,activation="relu"),
	tf.keras.layers.MaxPooling2D((2, 2)),
	tf.keras.layers.Dropout(0.2),

	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(64, activation="relu"),
	tf.keras.layers.Dense(10, activation='softmax')])
model1.compile(
	loss="sparse_categorical_crossentropy",
	optimizer = tf.keras.optimizers.Adam(),
	metrics =["accuracy"]
)

history = model1.fit(X_train,y_train,epochs=30,validation_data=(X_valid,y_valid))

