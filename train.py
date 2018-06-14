from sample_models import own_model
from train_utils import train_my_model

model = own_model(input_dim=161, output_dim=29)
train_my_model(model, pickle_path='own_model_loss.pickle', save_model_path='own_model.h5')