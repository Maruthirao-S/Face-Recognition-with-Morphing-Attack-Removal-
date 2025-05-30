from Evaluation import evaluation
from tensorflow import keras
from keras import layers




# Build the Transformer model with VGG and Attention
def transformer_vgg(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # VGG-like architecture
    vgg_base = keras.applications.VGG16(input_shape=input_shape, include_top=False, weights=None)(inputs)
    flattened = layers.Flatten()(vgg_base)

    # Self-attention mechanism
    attention = layers.Attention()([flattened, flattened])

    # Fully connected layers
    dense = layers.Dense(512, activation='relu')(attention)
    outputs = layers.Dense(num_classes, activation='softmax')(dense)

    model = keras.Model(inputs, outputs)
    return model


def Model_TVGG16_AM(train_data, train_target, test_data, test_target):

    # Define input shape and number of classes
    input_shape = (128, 128, 3)
    num_classes = train_target.shape[1]

    # Create the model
    model = transformer_vgg(input_shape, num_classes)
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_target, epochs=10, batch_size=32, validation_split=0.2)

    # Predict using test data
    predictions = model.predict(test_data).astype('int')

    # Save the model using TensorFlow's method
    model.save('transformer_vgg_model_1.h5')

    # # Load the model using TensorFlow's method
    # loaded_model = keras.models.load_model('transformer_vgg_model_.h5')

    Eval = evaluation(predictions, test_target)
    return predictions, Eval


