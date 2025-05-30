from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from Evaluation import evaluation

def Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target):
    # Normalize data
    Train_Data = Train_Data.astype('float32') / 255.0
    Test_Data = Test_Data.astype('float32') / 255.0
    num_classes = len(set(Train_Target))
    # Build the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=Train_Data.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(Train_Data, Train_Target, epochs=10, batch_size=32, validation_split=0.1)
    pred = model.predict(Test_Data)
    Eval = evaluation(pred,Test_Target)
    return Eval
