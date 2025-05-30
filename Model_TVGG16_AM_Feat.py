from tensorflow import keras
import numpy as np
from keras import layers
from keras import backend as K

def build_feature_extractor_with_attention(input_shape, attention_layer_name):
    inputs = keras.Input(shape=input_shape)

    vgg_base = keras.applications.VGG16(input_shape=input_shape, include_top=False, weights=None)(inputs)
    attention_layer_output = vgg_base  # Modify this line based on your model's structure

    # Create a model that outputs the activations of the specified layer with attention
    feature_extractor = keras.Model(inputs, attention_layer_output)

    return feature_extractor

def Model_TVGG16_AM_Feat(Images):
    input_shape = (128, 128, 3)

    # Use the build_feature_extractor_with_attention function to build the model
    layer_name_to_extract = 'attention'  # Replace with the actual layer name
    feature_extractor_with_attention = build_feature_extractor_with_attention(input_shape, layer_name_to_extract)
    outputs = [layer.output for layer in feature_extractor_with_attention.layers]  # all layer outputs
    inp = feature_extractor_with_attention.input
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    data = np.append(Images, Images, axis=0)
    Feats = []
    for i in range(data.shape[0]):
        test = data[i, :, :][np.newaxis, ...]
        layer_out = np.asarray(functors[2]([test])).squeeze()
        Feats.append(layer_out)
    Feature = np.asarray(Feats)
    return Feature

