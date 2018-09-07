from keras import backend as K
from keras.models import Model
from keras.activations import relu
from.keras.optimizers import Nadam
from keras.layers import BatchNormalization, Conv1D, Dense, Input, TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Lambda, ZeroPadding1D, RepeatVector

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = LSTM(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    if recur_layers == 1:
        layer = LSTM(units, return_sequences=True, activation='relu')(input_data)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = LSTM(units, return_sequences=True, activation='relu')(input_data)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(recur_layers - 2):
            layer = LSTM(units, return_sequences=True, activation='relu')(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(2+i))(layer)

        layer = LSTM(units, return_sequences=True, activation='relu')(layer)
        layer = BatchNormalization(name='bt_rnn_last_rnn')(layer)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, activation='relu'), merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=2, 
    cell=GRU, activation='tanh'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)


    if number_of_layers == 1:
        layer = cell(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = cell(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate)(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)
    

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

'''
 Model built by Gabriel Freire based on this paper https://arxiv.org/pdf/1512.02595v1.pdf
'''
def own_speech_to_text_model(input_dim, output_dim=29,dropout_rate=1, filters=200, rnn_size=256, kernel_size=11, strides=2):
    
    # Convolution configuration
    padding = 'valid'
    initialization = 'glorot_uniform'
    conv1d_config = {
        filters: filters,
        kernel_size: kernel_size,
        strides: strides,
        padding: padding,
        activation: 'relu',
        dilation_rate: 1
    }
    rnn_config = {
        units: rnn_size,
        activation: 'relu',
        return_sequences: True,
        implementation: 2,
        dropout_rate: dropout_rate,
        kernel_initializer: initialization
    }
    # Input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # 1 1D Convolutional layers
    conv1d_1 = Conv1D(**conv1d_config, name='conv1d_1')(input_data)
    conv_bn = BatchNormalization(name='conv1d_bn')(conv1d_1)
    
    # 7 Recurrent GRU Bidirectional LayersLayers
    gru_layer = GRU(**rnn_config, name='rnn_1')(conv_bn)
    gru_layer = BatchNormalization(name='bn_rnn_1')(gru_layer)
    
    for n in range(1):
        gru_layer = GRU(**rnn_config, name='rnn_{}'.format(n + 2))(gru_layer)
        gru_layer = BatchNormalization(name='bn_rnn_{}'.format(n + 2))(gru_layer)
    
    gru_layer = Bidirectional(GRU(**rnn_config, name='rnn_final'), merge_mode='concat')(gru_layer)
    gru_layer = BatchNormalization(name='bn_rnn_final')(gru_layer)
    
    # 1 Fully connected Layer
    time_dense = TimeDistributed(Dense(output_dim))(gru_layer)
    output_layer = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs= input_data, outputs=output_layer)
    
    # Specify dynamic output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, padding, strides)
    print(model.summary())
    return model

def own_grapheme_to_phoneme_model (layers, chars=29, phons=75, word_len=28, phon_len=28, tables=None,
        build=True, build_args=None, optimization=2):
    """
    Grapheme-to-phoneme converter; RNN GRU encoder-decoder model.
    # Arguments
        layers: Amount of layers for the encoder and decoder.
        chars: The amount of characters (English has 29).
        phons: The amount of phonemes (CMUDict uses 75).
        word_len: The length of the input word (CMUDict uses 28).
        phon_len: The length of the output phoneme (CMUDict uses 28).
        tables: Charecter en/decoding tables, can be retrieved by `get_cmudict()`.
        build: If to compile the model in Keras (the model will expect sprase labels).
    # Output
        A Keras model.
        Input:  `(word_length, chars)` shaped one-hot vectors.
        Output: `(word_length, phons)` shaped one-hot vectors.
    # Example
        ```
        (X_train, y_train), (X_test, y_test), (xtable, ytable) = get_cmudict()
        y_train = sparse_labels(y_train)
        model = G2P(layers=3, tables=(xtable, ytable))
        model.fit(X_train, y_train, batch_size=1024, epochs=20)
        ```
    """
    # V TODO: Engineer encoder model.
    # V TODO: Engineer decoder model.
    # V TODO: Get the state of an encoder's layer symbolically.
    # V TODO: Initialize a decoder layer state with the corresponding encoder layer state.
    # V TODO: Engineer the initial decoder input token (the output of encoder?).
    # V TODO: Feed the output of the decoder at t-1 as input at t.
    #   TODO: Engineer teacher forcing.

    # Decode data into neat named variables.
    if tables is not None:
        chars = len(tables[0].chars)
        phons = len(tables[1].chars)
        word_length = tables[0].maxlen
        phon_length = tables[1].maxlen

    # Define general RNN config.
    rnn_conf = {'units': phons,
                'return_sequences': True,
                'implementation': optimization}

    # Define our model's input.
    input_seq = Input((word_length, chars))

    ''' ENCODER '''
    # Multi-layer bidirectional GRU.
    # Keep an array of the encoder forward layer states to later (symbolically) initialize the decoder layers.
    # Define and add the encoders into the graph.
    encoder_conf = {**rnn_conf,
                    'return_state': True}
    encoder_bi_merge = 'sum'
    encoder_forward_states = [None]*layers

    encoded, encoder_forward_states[0], _ = Bidirectional(GRU(**encoder_conf), encoder_bi_merge)(input_seq)
    for layer in range(layers-2):
        encoded, encoder_forward_states[layer+1], _ = Bidirectional(GRU(**encoder_conf), encoder_bi_merge)(encoded)
    encoder_conf['return_sequences'] = False
    encoded, encoder_forward_states[-1], _ = Bidirectional(GRU(**encoder_conf), encoder_bi_merge)(encoded)
    assert not (None in encoder_forward_states), 'All encoder layer states haves to be assigned.'

    # Assign the encoder's final output as the decoder's initial input.
    # The encoder's output is of shape: `(phones)`.
    # The decoder expects input of shape: `(timestep, phones)`.
    # Use RV to add one timstep dimension to the encoder's output shape.
    input_decoder = RepeatVector(1)(encoded)

    # Teacher forcing.
    # ground_truth = Input((phon_len, phons))

    ''' DECODER '''
    # Multi-layer unidirectional GRU.
    # Initialize the decoder's layer states with the corresponding forward layer states from the encoder.
    # Define and add the decoders into the graph.
    decoder_conf = {**rnn_conf,
                    'unroll': True,
                    'output_length': phon_length}

    decoded = GRU(**decoder_conf)(input_decoder, initial_states=encoder_forward_states[0])
    for layer in range(layers-1):
        decoded = GRU(**decoder_conf)(decoded, initial_states=encoder_forward_states[layer+1])

    # Add a dense layer at each timestep to determine the output phonemes.
    # It will result in `(timesteps, number_of_phonemes)` shaped output values.
    output_densed = TimeDistributed(Dense(phons))(decoded)

    # Softmax to result in `(timesteps, number_of_phonemes)` shaped output probabilities.
    output_softmax = Activation('softmax')(output_densed)

    # Finalize the G2P model.
    g2p = Model(input_seq, output_softmax)

    if build:
        if build_args is None:
            build_args = {}
        if 'loss' not in build_args:
            build_args['loss'] = 'sparse_categorical_crossentropy'
        if 'optimizer' not in build_args:
            build_args['optimizer'] = Nadam()
        if 'metrics' not in build_args:
            build_args['metrics'] = ['accuracy']
        g2p.compile(**build_args)

    return g2p
