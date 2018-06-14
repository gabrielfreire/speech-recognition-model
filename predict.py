import numpy as np
from data_generator import AudioGenerator, vis_train_features, plot_spectrogram_feature, plot_mfcc_feature, plot_raw_audio
from sample_models import final_model, own_model
from keras import backend as K
from keras.layers import (GRU)
from utils import int_sequence_to_text
from text import correction
from IPython.display import Audio
import os

def plot_audio_visualizations(index=0):  
    # plot audio visualizations
    vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path = vis_train_features(index=index)
    plot_spectrogram_feature(vis_spectrogram_feature)
    plot_mfcc_feature(vis_mfcc_feature)
    plot_raw_audio(vis_raw_audio)

def get_predictions(index, partition, trained_model, model_path):
    """ Print a model's decoded predictions
    Params:
        index (int): The example you would like to visualize
        partition (str): One of 'train' or 'validation'
        trained_model (Model): The acoustic model
        model_path (str): Path to saved acoustic model's weights
    """
    # load the train and test data
    data_gen = AudioGenerator(spectrogram=False)
    data_gen.load_train_data()
    data_gen.load_validation_data()
        
    # obtain the true transcription and the audio features from Dataset
    if partition == 'validation':
        transcr = data_gen.valid_texts[index]
        audio_path = data_gen.valid_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    elif partition == 'train':
        transcr = data_gen.train_texts[index]
        audio_path = data_gen.train_audio_paths[index]
        data_point = data_gen.normalize(data_gen.featurize(audio_path))
    else:
        raise Exception('Invalid partition!  Must be "train" or "validation"')

    print("Trained model output length:\n" + str(trained_model.output_length(data_point.shape[0])))
    # obtain and decode the acoustic model's predictions
    trained_model.load_weights(model_path)
    prediction = trained_model.predict(np.expand_dims(data_point, axis=0))
    output_length = [trained_model.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                        prediction, output_length)[0][0])+1).flatten().tolist()
    
    transcription = ''.join(int_sequence_to_text(pred_ints))
    # Correction using KenLM language model toolkit
    corrected_transcription = correction(transcription)
 
    print('-'*80)
    print(repr(audio_path).replace(r"\\", r"/"))
    print('True transcription:\n' + '\n' + transcr)
    print('-'*80)
    print('Raw prediction:\n' + str(prediction[0]))
    print('CTC Decoded predicted Ints before conversion to text:\n' + str(pred_ints))
    print('Predicted transcription:\n' + '\n' + transcription)
    print('Predicted transcription with correction:\n' + corrected_transcription)
    print('-'*80)

"""
 Gabriel Freire: My final compiled Model
 Optimizer: SGD
 Loss: CTC
 file: sample_models.py
"""
my_model = own_model(input_dim=161, output_dim=29)
get_predictions(index=0, partition='train', trained_model=my_model, model_path='results/own_model.h5')