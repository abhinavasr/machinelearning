#We will use Pandas to read the csv file
import pandas as pd

file1 = "../data/input_rand1.csv"

# Incase you have mode than 1 csv, you may want to use this piece of code to combine them
all_files = [file1]
dataset = pd.concat((pd.read_csv(f,delimiter=',') for f in all_files))

# We don't need empty values
dataset = dataset.dropna(how="any", axis=0)

#replace spaces with _ for headers
dataset.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)

train = dataset.sample(frac=0.8,random_state=200)
test = dataset.drop(train.index)

X_train = train.drop("Y", axis=1)
Y_train = train["Y"]

validation = train #test.iloc[[1]]
#test.drop(test.index[1], inplace=True)

X_test = test.drop("Y", axis=1)
Y_test = test["Y"]

from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "Linear_{}".format(int(time.time()))

tb = TensorBoard(log_dir='logs/{}'.format(NAME))

model = keras.Sequential([
    keras.layers.Dense(10, input_shape=[len(list(X_test))]),
    #keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

model.compile(loss='mse',
                optimizer='adam',
                metrics=['mae', 'mse'])

model.summary()

model.fit(X_train, Y_train, epochs=1000, validation_split = 0.2, callbacks=[tb])
loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)

#predicting

input_dict = train
input_dict_x = input_dict.drop("Y", axis=1)
input_dict_y = input_dict["Y"]

predict_results = model.predict(input_dict_x)
print(predict_results)

predictions_Original  = predict_results


# New Dataset
d = {'X': [7,8,9,10,11]}
df = pd.DataFrame(data=d)
predict_results = model.predict(df)

print(predict_results)

new_predections = predict_results

from plotter import plot_graph

plot_graph(train, predictions_Original, new_predections,"graph_{}".format(int(time.time())))