import numpy as np
import cv2 as cv
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from Evaluation import evaluation

def Model_DCNN(train_data, Y, test_data, test_y):
    IMG_SIZE = 20

    # Resize and normalize input data
    def preprocess(data):
        resized = np.zeros((len(data), IMG_SIZE, IMG_SIZE))
        for i in range(len(data)):
            resized[i] = cv.resize(data[i].astype('float32'), (IMG_SIZE, IMG_SIZE))
        return np.expand_dims(resized, axis=-1)  # Add channel dimension

    X_train = preprocess(train_data)
    X_test = preprocess(test_data)

    # Define model using Keras Functional API
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input')
    x = Conv2D(32, kernel_size=5, activation='relu', name='conv1')(inputs)
    x = MaxPooling2D(pool_size=2, name='pool1')(x)

    x = Conv2D(64, kernel_size=5, activation='relu', name='conv2')(x)
    x = MaxPooling2D(pool_size=2, name='pool2')(x)

    x = Conv2D(128, kernel_size=3, activation='relu', name='conv3')(x)
    x = MaxPooling2D(pool_size=2, name='pool3')(x)

    x = Conv2D(64, kernel_size=3, activation='relu', name='conv4')(x)
    x = MaxPooling2D(pool_size=2, name='pool4')(x)

    x = Conv2D(32, kernel_size=3, activation='relu', name='conv5')(x)
    x = MaxPooling2D(pool_size=2, name='pool5')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.2)(x)  # Keep probability of 0.8

    outputs = Dense(Y.shape[1], activation='linear', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=SGD(learning_rate=0.01), loss='mean_squared_error', metrics=['mae'])

    # Train the model
    model.fit(X_train, Y, epochs=10, batch_size=32,
              validation_data=(X_test, test_y), verbose=1)

    # Predict and evaluate
    predictions = model.predict(X_test)
    predictions = np.round(np.abs(predictions)).astype('int')
    Eval = evaluation(predictions, test_y)
    return np.asarray(Eval).ravel(), predictions
