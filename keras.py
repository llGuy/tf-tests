import keras

# Declare our shit
batch_size = 128

num_classes = 10
epochs = 10

img_x, img_y = 28, 28

# Guts of the Keras code:
model = Sequential()

# We have 32 output channels, 5x5 moving filter / window, strides, ...
model.add(Conv2D(32, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', input_shape = input_shape))

model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(64, (5, 5), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(1000, activation = 'relu'))

model.add(Dense(num_classes, activation = 'softmax'))


