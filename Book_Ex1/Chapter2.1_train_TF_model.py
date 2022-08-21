import pandas as pd
import tensorflow as tf




housing_prepared = pd.read_csv(r"C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\Book_Ex1\housing_prepared.csv")
housing_labels = pd.to_csv(r"C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\Book_Ex1\housing_labels.csv")




model_7 = tf.keras.Sequential([
	tf.keras.layers.Dense(150,activation="tanh"),
	tf.keras.layers.Dense(150,activation="tanh"),
	tf.keras.layers.Dense(150,activation="tanh"),
	tf.keras.layers.Dense(1,activation="sigmoid")
])

#2 Compile the model
model_7.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(),
	metrics=["mae"])
 
#2.5 Create a learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*10**(epoch/20))

#3.Fit the model
history_7 = model_7.fit(tf_housing_prepared,tf_housing_labels,epochs=200,callbacks=[lr_scheduler], verbose=1)

print(f"Evaluación de modelo 7: {model_7.evaluate(tf_housing_prepared,tf_housing_labels)}")

history_7_df = pd.DataFrame(history_7.history)

#model_7.save(resource_path(r"save_model_Non_Linear"))
#plot_decision_boundary(model_7,X,y)

temp_model_7_df = history_7_df.drop("accuracy",axis=1)
temp_model_7_df.plot()
plt.show()