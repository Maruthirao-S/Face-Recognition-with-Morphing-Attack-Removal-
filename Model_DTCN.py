import numpy as np
from keras import Input
from keras.src.layers import Conv1D, Dropout, Flatten, Dense

from Evaluation import evaluation
from tensorflow import keras


def Model_DTCN(train_data, train_target, test_data, test_target, sol):
    input_length = train_data.shape[1]
    num_classes = train_target.shape[1]
    train_data = np.expand_dims(train_data, axis=2)

    # Build DeepTCN model
    input_shape = (input_length, 1)
    num_filters = sol[0].astype('int')
    kernel_size = 5

    input_layer = Input(shape=input_shape)
    x = input_layer

    for _ in range(4):  # Number of convolutional blocks
        x = Conv1D(num_filters, kernel_size, activation='relu')(x)
        x = Dropout(0.2)(x)

    x = Flatten()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = keras.Model(input_layer, output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_target, epochs=sol[1], batch_size=64, steps_per_epoch = sol[2].astype('int'))

    Predict = model.predict(test_data)

    # Find the maximum value in each row
    max_values = Predict.max(axis=1, keepdims=True)

    # Create a categorical array where the maximum value is 1 and others are 0
    categorical_array = (Predict == max_values)

    Eval = evaluation(int(categorical_array), test_target)
    return Eval
