import tensorflow as tf

print(tf.__version__)

if tf.test.gpu_device_name(): 

    print(f"Default GPU Device:{tf.test.gpu_device_name()}")

else:

   print("Please install GPU version of TF")