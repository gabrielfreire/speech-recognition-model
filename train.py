from sample_models import own_model
from train_utils import train_my_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf 
# allocate 50% of GPU memory (if you like, feel free to change this)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# train
model = own_model(input_dim=161, output_dim=29)
train_my_model(model, pickle_path='own_model_loss.pickle', save_model_path='own_model.h5')