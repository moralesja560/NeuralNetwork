from cmath import sin
import tensorflow as tf
import numpy as np
import os
import sys
import math
import matplotlib.pyplot as plt

#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

tf.random.set_seed(42)
X = np.array([-100,-99.6,-99.2,-98.8,-98.4,-98,-97.6,-97.2,-96.8,-96.399,-95.999,-95.599,-95.199,-94.799,-94.399,-93.999,-93.599,-93.199,-92.799,-92.399,-91.999,-91.599,-91.199,-90.799,-90.399,-89.999,-89.599,-89.199,-88.799,-88.399,-87.999,-87.599,-87.199,-86.799,-86.399,-85.999,-85.599,-85.199,-84.799,-84.399,-83.999,-83.599,-83.199,-82.799,-82.399,-81.999,-81.599,-81.199,-80.799,-80.399,-79.999,-79.599,-79.199,-78.799,-78.399,-77.999,-77.599,-77.199,-76.799,-76.399,-75.999,-75.599,-75.199,-74.799,-74.399,-73.999,-73.599,-73.199,-72.799,-72.399,-71.999,-71.599,-71.199,-70.799,-70.399,-69.999,-69.599,-69.199,-68.799,-68.399,-67.999,-67.599,-67.199,-66.799,-66.399,-65.999,-65.599,-65.199,-64.799,-64.399,-63.999,-63.599,-63.199,-62.799,-62.399,-61.999,-61.599,-61.199,-60.799,-60.399,-59.999,-59.599,-59.199,-58.799,-58.399,-57.999,-57.599,-57.199,-56.799,-56.399,-55.999,-55.599,-55.199,-54.799,-54.399,-53.999,-53.599,-53.199,-52.799,-52.399,-51.999,-51.599,-51.199,-50.799,-50.399,-49.999,-49.599,-49.199,-48.799,-48.399,-47.999,-47.599,-47.199,-46.799,-46.399,-45.999,-45.599,-45.199,-44.799,-44.399,-43.999,-43.599,-43.199,-42.799,-42.399,-41.999,-41.599,-41.199,-40.799,-40.399,-39.999,-39.599,-39.199,-38.799,-38.399,-37.999,-37.599,-37.199,-36.799,-36.399,-35.999,-35.599,-35.199,-34.799,-34.399,-33.999,-33.599,-33.199,-32.799,-32.399,-31.999,-31.599,-31.199,-30.799,-30.399,-29.999,-29.599,-29.199,-28.799,-28.399,-27.999,-27.599,-27.199,-26.799,-26.399,-25.999,-25.599,-25.199,-24.799,-24.399,-23.999,-23.599,-23.199,-22.799,-22.399,-21.999,-21.599,-21.199,-20.799,-20.399,-19.999,-19.599,-19.199,-18.799,-18.399,-17.999,-17.599,-17.199,-16.799,-16.399,-15.999,-15.599,-15.199,-14.799,-14.399,-13.999,-13.599,-13.199,-12.799,-12.399,-11.999,-11.599,-11.199,-10.799,-10.399,-9.9999,-9.5999,-9.1999,-8.7999,-8.3999,-7.9999,-7.5999,-7.1999,-6.7999,-6.3999,-5.9999,-5.5999,-5.1999,-4.7999,-4.3999,-3.9999,-3.5999,-3.1999,-2.7999,-2.3999,-1.9999,-1.5999,-1.1999,-0.7999,-0.3999,3.45390,0.40000,0.80000,1.20000,1.60000,2.00000,2.40000,2.80000,3.20000,3.60000,4.00000,4.40000,4.80000,5.20000,5.60000,6.00000,6.40000,6.80000,7.20000,7.60000,8.00000,8.40000,8.80000,9.20000,9.60000,10.0000,10.4000,10.8000,11.2000,11.6000,12.0000,12.4000,12.8000,13.2000,13.6000,14.0000,14.4000,14.8000,15.2000,15.6000,16.0000,16.4000,16.8000,17.2000,17.6000,18.0000,18.4000,18.8000,19.2000,19.6000,20.0000,20.4000,20.8000,21.2000,21.6000,22.0000,22.4000,22.8000,23.2000,23.6000,24.0000,24.4000,24.8000,25.2000,25.6000,26.0000,26.4000,26.8000,27.2000,27.6000,28.0000,28.4000,28.8000,29.2000,29.6000,30.0000,30.4000,30.8000,31.2000,31.6000,32.0000,32.4000,32.8000,33.2000,33.6000,34.0000,34.4000,34.8000,35.2000,35.6000,36.0000,36.4000,36.8000,37.2000,37.6000,38.0000,38.4000,38.8000,39.2000,39.6000,40.0000,40.4000,40.8000,41.2000,41.6000,42.0000,42.4000,42.8000,43.2000,43.6000,44.0000,44.4000,44.8000,45.2000,45.6000,46.0000,46.4000,46.8000,47.2000,47.6000,48.0000,48.4000,48.8000,49.2000,49.6000,50.0000,50.4000,50.8000,51.2000,51.6000,52.0000,52.4000,52.8000,53.2000,53.6000,54.0000,54.4000,54.8000,55.2000,55.6000,56.0000,56.4000,56.8000,57.2000,57.6000,58.0000,58.4000,58.8000,59.2000,59.6000,60.0000,60.4000,60.8000,61.2000,61.6000,62.0000,62.4000,62.8000,63.2000,63.6000,64.0000,64.4000,64.8000,65.2000,65.6000,66.0000,66.4000,66.8000,67.2000,67.6000,68.0000,68.4000,68.8000,69.2000,69.6000,70.0000,70.4000,70.8000,71.2000,71.6000,72.0000,72.4000,72.8000,73.2000,73.6000,74.0000,74.4000,74.8000,75.2000,75.6000,76.0000,76.4000,76.8000,77.2000,77.6000,78.0000,78.4000,78.8000,79.2000,79.6000,80.0000,80.4000,80.8000,81.2000,81.6000,82.0000,82.4000,82.8000,83.2000,83.6000,84.0000,84.4000,84.8000,85.2000,85.6000,86.0000,86.4000,86.8000,87.2000,87.6000,88.0000,88.4000,88.8000,89.2000,89.6000,90.0000,90.4000,90.8000,91.2000,91.6000,92.0000,92.4000,92.8000,93.2000,93.6000,94.0000,94.4000,94.8000,95.2000,95.6000,96.0000,96.4000,96.8000,97.2000,97.6000,98.0000,98.4000,98.8000,99.2000,99.6000,100.000])
#y = np.array([0.50636,0.80219,0.97137,0.98720,0.84716,0.57338,0.20907,-0.1882,-0.5558,-0.8356,-0.9835,-0.9762,-0.8147,-0.5245,-0.1516,0.24525,0.60341,0.86631,0.99244,0.96188,0.77946,0.47398,9.36755,-0.3014,-0.6489,-0.8939,-0.9979,-0.9442,-0.7415,-0.4217,-3.5398,0.35657,0.69224,0.91862,0.99998,0.92345,0.70114,0.36813,-2.2999,-0.4104,-0.7331,-0.9401,-0.9986,-0.8994,-0.6583,-0.3132,8.13191,0.46302,0.77163,0.95841,0.99388,0.87244,0.61326,0.25725,-0.1393,-0.5139,-0.8074,-0.9734,-0.9857,-0.8424,-0.5661,-0.2004,0.19692,0.56317,0.84050,0.98514,0.97424,0.80954,0.51702,0.14287,-0.2538,-0.6104,-0.8707,-0.9934,-0.9594,-0.7738,-0.4661,-8.4859,0.30985,0.65564,0.89792,0.99844,0.94133,0.73560,0.41373,2.65511,-0.3648,-0.6986,-0.9220,-0.9999,-0.9200,-0.6948,-0.3598,3.18476,0.41855,0.73918,0.94310,0.99813,0.89558,0.65163,0.30481,-9.0137,-0.4708,-0.7772,-0.9609,-0.9928,-0.8680,-0.6062,-0.2486,0.14812,0.52155,0.81263,0.97543,0.98422,0.83762,0.55878,0.19173,-0.2055,-0.5704,-0.8452,-0.9866,-0.9722,-0.8043,-0.5094,-0.1341,0.26237,0.61743,0.87502,0.99445,0.95689,0.76825,0.45832,7.60367,-0.3182,-0.6623,-0.9017,-0.9989,-0.9383,-0.7295,-0.4056,-1.7701,0.37305,0.70491,0.92547,0.99993,0.91652,0.68841,0.35161,-4.0693,-0.4265,-0.7451,-0.9460,-0.9975,-0.8916,-0.6448,-0.2963,9.89496,0.47864,0.78277,0.96332,0.99177,0.86365,0.59918,0.24011,-0.1568,-0.5290,-0.8177,-0.9773,-0.9826,-0.8327,-0.5514,-0.1830,0.21425,0.57771,0.84996,0.98803,0.97010,0.79902,0.50178,0.12533,-0.2709,-0.6243,-0.8792,-0.9953,-0.9542,-0.7625,-0.4504,-6.7208,0.32663,0.66890,0.90557,0.99927,0.93520,0.72349,0.39755,8.85130,-0.3812,-0.7111,-0.9287,-0.9997,-0.9129,-0.6819,-0.3433,4.95356,0.43456,0.75098,0.94884,0.99690,0.88756,0.63810,0.28790,-0.1077,-0.4863,-0.7882,-0.9656,-0.9906,-0.8591,-0.5920,-0.2315,0.16560,0.53657,0.82282,0.97917,0.98093,0.82782,0.54402,0.17432,-0.2228,-0.5849,-0.8545,-0.9893,-0.9679,-0.7936,-0.4941,-0.1165,0.27941,0.63126,0.88345,0.99616,0.95160,0.75680,0.44252,5.83741,-0.3349,-0.6754,-0.9092,-0.9995,-0.9320,-0.7173,-0.3894,3.45390,0.38941,0.71735,0.93203,0.99957,0.90929,0.67546,0.33498,-5.8374,-0.4425,-0.7568,-0.9516,-0.9961,-0.8834,-0.6312,-0.2794,0.11654,0.49411,0.79366,0.96791,0.98935,0.85459,0.58491,0.22288,-0.1743,-0.5440,-0.8278,-0.9809,-0.9791,-0.8228,-0.5365,-0.1656,0.23150,0.59207,0.85916,0.99060,0.96565,0.78825,0.48639,0.10775,-0.2879,-0.6381,-0.8875,-0.9969,-0.9488,-0.7509,-0.4345,-4.9535,0.34331,0.68196,0.91294,0.99979,0.92879,0.71116,0.38125,-8.8513,-0.3975,-0.7234,-0.9352,-0.9992,-0.9055,-0.6689,-0.3266,6.72080,0.45044,0.76255,0.95428,0.99535,0.87927,0.62437,0.27090,-0.1253,-0.5017,-0.7990,-0.9701,-0.9880,-0.8499,-0.5777,-0.2142,0.18303,0.55142,0.83275,0.98261,0.97734,0.81776,0.52908,0.15686,-0.2401,-0.5991,-0.8636,-0.9917,-0.9633,-0.7827,-0.4786,-9.8949,0.29636,0.64489,0.89160,0.99755,0.94601,0.74511,0.42657,4.06932,-0.3516,-0.6884,-0.9165,-0.9999,-0.9254,-0.7049,-0.3730,1.77019,0.40566,0.72957,0.93830,0.99890,0.90178,0.66230,0.31825,-7.6036,-0.4583,-0.7682,-0.9568,-0.9944,-0.8750,-0.6174,-0.2623,0.13411,0.50942,0.80431,0.97221,0.98662,0.84527,0.57046,0.20559,-0.1917,-0.5587,-0.8376,-0.9842,-0.9754,-0.8126,-0.5215,-0.1481,0.24869,0.60624,0.86808,0.99287,0.96090,0.77723,0.47085,9.01379,-0.3048,-0.6516,-0.8955,-0.9981,-0.9431,-0.7391,-0.4185,-3.1847,0.35988,0.69480,0.92002,0.99999,0.92208,0.69860,0.36482,-2.6551,-0.4137,-0.7356,-0.9413,-0.9984,-0.8979,-0.6556,-0.3098,8.48594,0.46617,0.77389,0.95942,0.99349,0.87070,0.61045,0.25382,-0.1428,-0.5170,-0.8095,-0.9742,-0.9851,-0.8405,-0.5631,-0.1969,0.20040,0.56610,0.84242,0.98575,0.97344,0.80744,0.51397,0.13936,-0.2572,-0.6132,-0.8724,-0.9938,-0.9584,-0.7716,-0.4630,-8.1319,0.31322,0.65832,0.89948,0.99863,0.94012,0.73319,0.41049,2.29996,-0.3681,-0.7011,-0.9234,-0.9999,-0.9186,-0.6922,-0.3565,3.53983,0.42177,0.74156,0.94428,0.99791,0.89399,0.64893,0.30142,-9.3675,-0.4739,-0.7794,-0.9618,-0.9924,-0.8663,-0.6034,-0.2452,0.15163,0.52457,0.81470,0.97620,0.98358,0.83568,0.55583,0.18824,-0.2090,-0.5733,-0.8471,-0.9872,-0.9713,-0.8021,-0.5063])
y = np.array([10003,9923.16,9843.64,9764.44,9685.56,9606.99,9528.75,9450.83,9373.23,9295.95,9218.99,9142.35,9066.03,8990.03,8914.35,8838.99,8763.95,8689.23,8614.83,8540.75,8466.99,8393.55,8320.43,8247.63,8175.15,8102.99,8031.15,7959.63,7888.43,7817.55,7746.99,7676.75,7606.83,7537.23,7467.95,7398.99,7330.35,7262.03,7194.03,7126.35,7058.99,6991.95,6925.23,6858.83,6792.75,6726.99,6661.55,6596.43,6531.63,6467.15,6402.99,6339.15,6275.63,6212.43,6149.55,6086.99,6024.75,5962.83,5901.23,5839.95,5778.99,5718.35,5658.03,5598.03,5538.35,5478.99,5419.95,5361.23,5302.83,5244.75,5186.99,5129.55,5072.43,5015.63,4959.15,4902.99,4847.15,4791.63,4736.43,4681.55,4626.99,4572.75,4518.83,4465.23,4411.95,4358.99,4306.35,4254.03,4202.03,4150.35,4098.99,4047.95,3997.23,3946.83,3896.75,3846.99,3797.55,3748.43,3699.63,3651.15,3602.99,3555.15,3507.63,3460.43,3413.55,3366.99,3320.75,3274.83,3229.23,3183.95,3138.99,3094.35,3050.03,3006.03,2962.35,2918.99,2875.95,2833.23,2790.83,2748.75,2706.99,2665.55,2624.43,2583.63,2543.15,2502.99,2463.15,2423.63,2384.43,2345.55,2306.99,2268.75,2230.83,2193.23,2155.95,2118.99,2082.35,2046.03,2010.03,1974.35,1938.99,1903.95,1869.23,1834.83,1800.75,1766.99,1733.55,1700.43,1667.63,1635.15,1602.99,1571.15,1539.63,1508.43,1477.55,1446.99,1416.75,1386.83,1357.23,1327.95,1298.99,1270.35,1242.03,1214.03,1186.35,1158.99,1131.95,1105.23,1078.83,1052.75,1026.99,1001.55,976.439,951.639,927.159,902.999,879.159,855.639,832.439,809.559,786.999,764.759,742.839,721.239,699.959,678.999,658.359,638.039,618.039,598.359,578.999,559.959,541.239,522.839,504.759,486.999,469.559,452.439,435.639,419.159,402.999,387.159,371.639,356.439,341.559,326.999,312.759,298.839,285.239,271.959,258.999,246.359,234.039,222.039,210.359,198.999,187.959,177.239,166.839,156.759,146.999,137.559,128.439,119.639,111.159,102.999,95.1599,87.6399,80.4399,73.5599,66.9999,60.7599,54.8399,49.2399,43.9599,38.9999,34.3599,30.0399,26.0399,22.3599,18.9999,15.9599,13.2399,10.8399,8.75999,6.99999,5.55999,4.43999,3.63999,3.15999,3,3.16000,3.64000,4.44000,5.56000,7.00000,8.76000,10.8400,13.2400,15.9600,19.0000,22.3600,26.0400,30.0400,34.3600,39.0000,43.9600,49.2400,54.8400,60.7600,67.0000,73.5600,80.4400,87.6400,95.1600,103.000,111.160,119.640,128.440,137.560,147.000,156.760,166.840,177.240,187.960,199.000,210.360,222.040,234.040,246.360,259.000,271.960,285.240,298.840,312.760,327.000,341.560,356.440,371.640,387.160,403.000,419.160,435.640,452.440,469.560,487.000,504.760,522.840,541.240,559.960,579.000,598.360,618.040,638.040,658.360,679.000,699.960,721.240,742.840,764.760,787.000,809.560,832.440,855.640,879.160,903.000,927.160,951.640,976.440,1001.56,1027.00,1052.76,1078.84,1105.24,1131.96,1159.00,1186.36,1214.04,1242.04,1270.36,1299.00,1327.96,1357.24,1386.84,1416.76,1447.00,1477.56,1508.44,1539.64,1571.16,1603.00,1635.16,1667.64,1700.44,1733.56,1767.00,1800.76,1834.84,1869.24,1903.96,1939.00,1974.36,2010.04,2046.04,2082.36,2119.00,2155.96,2193.24,2230.84,2268.76,2307.00,2345.56,2384.44,2423.64,2463.16,2503.00,2543.16,2583.64,2624.44,2665.56,2707.00,2748.76,2790.84,2833.24,2875.96,2919.00,2962.36,3006.04,3050.04,3094.36,3139.00,3183.96,3229.24,3274.84,3320.76,3367.00,3413.56,3460.44,3507.64,3555.16,3603.00,3651.16,3699.64,3748.44,3797.56,3847.00,3896.76,3946.84,3997.24,4047.96,4099.00,4150.36,4202.04,4254.04,4306.36,4359.00,4411.96,4465.24,4518.84,4572.76,4627.00,4681.56,4736.44,4791.64,4847.16,4903.00,4959.16,5015.64,5072.44,5129.56,5187.00,5244.76,5302.84,5361.24,5419.96,5479.00,5538.36,5598.04,5658.04,5718.36,5779.00,5839.96,5901.24,5962.84,6024.76,6087.00,6149.56,6212.44,6275.64,6339.16,6403.00,6467.16,6531.64,6596.44,6661.56,6727.00,6792.76,6858.84,6925.24,6991.96,7059.00,7126.36,7194.04,7262.04,7330.36,7399.00,7467.96,7537.24,7606.84,7676.76,7747.00,7817.56,7888.44,7959.64,8031.16,8103.00,8175.16,8247.64,8320.44,8393.56,8467.00,8540.76,8614.84,8689.24,8763.96,8839.00,8914.36,8990.04,9066.04,9142.36,9219.00,9295.96,9373.24,9450.84,9528.76,9607.00,9685.56,9764.44,9843.64,9923.16,10003])
X = tf.constant(X)
y = tf.constant(y)

print(plt.scatter(X,y))
plt.show()

#Step 1.- Create a model using the Sequential API
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(100, activation ="relu"))
model.add(tf.keras.layers.Dense(100, activation =None))
model.add(tf.keras.layers.Dense(1))

# Step 2 .- Compile the model
#we can add as much layers as we want
model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.Adam(learning_rate= 0.01),
              metrics = ["mae"]
              )
#fit the model
model.fit(tf.expand_dims(X, axis=-1),y, epochs=800)


model.save(resource_path(r"savedmodel2"))

#predict data
input_value = 190

correct_value = (input_value*input_value) +3

neural_network_out = round(np.double(model.predict(np.array([input_value]))),5)
print(f"valor entrada: {input_value}, \n valor salida correcto {correct_value} \n red neuronal: {neural_network_out} \n diferencia: {round(correct_value - neural_network_out,5)}")
