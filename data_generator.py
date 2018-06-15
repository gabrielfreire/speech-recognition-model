"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

import json
import numpy as np
import random
from python_speech_features import mfcc
import librosa
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence
from utils import conv_output_length

RNG_SEED = 123

class AudioGenerator():
    def __init__(self, step=10, window=20, max_freq=8000, mfcc_dim=13,
        minibatch_size=20, desc_file=None, spectrogram=True, max_duration=10.0, 
        sort_by_duration=False):
        """
        Params:
            step (int): Step size in milliseconds between windows (for spectrogram ONLY)
            window (int): FFT window size in milliseconds (for spectrogram ONLY)
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned (for spectrogram ONLY)
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """

        self.feat_dim = calc_feat_dim(window, max_freq)
        self.mfcc_dim = mfcc_dim
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.rng = random.Random(RNG_SEED)
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.cur_train_index = 0
        self.cur_valid_index = 0
        self.cur_test_index = 0
        self.max_duration=max_duration
        self.minibatch_size = minibatch_size
        self.spectrogram = spectrogram
        self.sort_by_duration = sort_by_duration

    def get_batch(self, partition):
        """ Obtain a batch of train, validation, or test data
        """
        if partition == 'train':
            audio_paths = self.train_audio_paths
            cur_index = self.cur_train_index
            texts = self.train_texts
        elif partition == 'valid':
            audio_paths = self.valid_audio_paths
            cur_index = self.cur_valid_index
            texts = self.valid_texts
        elif partition == 'test':
            audio_paths = self.test_audio_paths
            cur_index = self.test_valid_index
            texts = self.test_texts
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")

        features = [self.normalize(self.featurize(a)) for a in 
            audio_paths[cur_index:cur_index+self.minibatch_size]]

        # calculate necessary sizes
        max_length = max([features[i].shape[0] 
            for i in range(0, self.minibatch_size)])
        max_string_length = max([len(texts[cur_index+i]) 
            for i in range(0, self.minibatch_size)])
        
        # initialize the arrays
        X_data = np.zeros([self.minibatch_size, max_length, 
            self.feat_dim*self.spectrogram + self.mfcc_dim*(not self.spectrogram)])
        labels = np.ones([self.minibatch_size, max_string_length]) * 28
        input_length = np.zeros([self.minibatch_size, 1])
        label_length = np.zeros([self.minibatch_size, 1])
        
        for i in range(0, self.minibatch_size):
            # calculate X_data & input_length
            feat = features[i]
            input_length[i] = feat.shape[0]
            X_data[i, :feat.shape[0], :] = feat

            # calculate labels & label_length
            label = np.array(text_to_int_sequence(texts[cur_index+i])) 
            labels[i, :len(label)] = label
            label_length[i] = len(label)
        # return the arrays
        outputs = {'ctc': np.zeros([self.minibatch_size])}
        inputs = {'the_input': X_data, 
                  'the_labels': labels, 
                  'input_length': input_length, 
                  'label_length': label_length 
                 }
        return (inputs, outputs)

    def shuffle_data_by_partition(self, partition):
        """ Shuffle the training or validation data
        """
        if partition == 'train':
            self.train_audio_paths, self.train_durations, self.train_texts = shuffle_data(
                self.train_audio_paths, self.train_durations, self.train_texts)
        elif partition == 'valid':
            self.valid_audio_paths, self.valid_durations, self.valid_texts = shuffle_data(
                self.valid_audio_paths, self.valid_durations, self.valid_texts)
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")

    def sort_data_by_duration(self, partition):
        """ Sort the training or validation sets by (increasing) duration
        """
        if partition == 'train':
            self.train_audio_paths, self.train_durations, self.train_texts = sort_data(
                self.train_audio_paths, self.train_durations, self.train_texts)
        elif partition == 'valid':
            self.valid_audio_paths, self.valid_durations, self.valid_texts = sort_data(
                self.valid_audio_paths, self.valid_durations, self.valid_texts)
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")

    def next_train(self):
        """ Obtain a batch of training data
        """
        while True:
            ret = self.get_batch('train')
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= len(self.train_texts) - self.minibatch_size:
                self.cur_train_index = 0
                self.shuffle_data_by_partition('train')
            yield ret    

    def next_valid(self):
        """ Obtain a batch of validation data
        """
        while True:
            ret = self.get_batch('valid')
            self.cur_valid_index += self.minibatch_size
            if self.cur_valid_index >= len(self.valid_texts) - self.minibatch_size:
                self.cur_valid_index = 0
                self.shuffle_data_by_partition('valid')
            yield ret

    def next_test(self):
        """ Obtain a batch of test data
        """
        while True:
            ret = self.get_batch('test')
            self.cur_test_index += self.minibatch_size
            if self.cur_test_index >= len(self.test_texts) - self.minibatch_size:
                self.cur_test_index = 0
            yield ret

    def load_train_data(self, desc_file='train_corpus.json'):
        self.load_metadata_from_desc_file(desc_file, 'train')
        self.fit_train()
        if self.sort_by_duration:
            self.sort_data_by_duration('train')

    def load_validation_data(self, desc_file='valid_corpus.json'):
        self.load_metadata_from_desc_file(desc_file, 'validation')
        if self.sort_by_duration:
            self.sort_data_by_duration('valid')

    def load_test_data(self, desc_file='test_corpus.json'):
        self.load_metadata_from_desc_file(desc_file, 'test')
    
    def load_metadata_from_desc_file(self, desc_file, partition='train'):
        """ Read metadata from a JSON-line file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
        """
        audio_paths, durations, texts = [], [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > self.max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                except Exception as e:
                    # Change to (KeyError, ValueError) or
                    # (KeyError,json.decoder.JSONDecodeError), depending on
                    # json module version
                    print('Error reading line #{}: {}'
                                .format(line_num, json_line))
        if partition == 'train':
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
        elif partition == 'validation':
            self.valid_audio_paths = audio_paths
            self.valid_durations = durations
            self.valid_texts = texts
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
        else:
            raise Exception("Invalid partition to load metadata. "
             "Must be train/validation/test")
            
    def fit_train(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.train_audio_paths))
        samples = self.rng.sample(self.train_audio_paths, k_samples)
        feats = [self.featurize(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)
        
    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the corresponding feature
        Params:
            audio_clip (str): Path to the audio clip
        """
        if self.spectrogram:
            return spectrogram_from_file(
                audio_clip, step=self.step, window=self.window,
                max_freq=self.max_freq)
        else:
            (rate, sig) = wav.read(audio_clip)
            return mfcc(sig, rate, numcep=self.mfcc_dim)

    def normalize(self, feature, eps=1e-14):
        """ Center a feature using the mean and std
        Params:
            feature (numpy.ndarray): Feature to normalize

            Mean values for a Spectrogram feature:
                np.array([-20.28586078 -18.08537968 -17.13736808 -16.47337494 -16.26820409
                    -16.55408883 -17.12967673 -17.35698312 -17.31574859 -17.29727512
                    -17.44904873 -17.74196771 -18.11406208 -18.53870863 -18.91410474
                    -19.1746619  -19.41891623 -19.66285232 -19.89427369 -20.11371117
                    -20.24873686 -20.36581912 -20.50921723 -20.60100762 -20.67430896
                    -20.73181712 -20.76904386 -20.72445087 -20.69483405 -20.75668029
                    -20.76588406 -20.77106311 -20.79673417 -20.8378663  -20.9099631
                    -20.96461743 -20.98967449 -21.04559159 -21.1083131  -21.16559797
                    -21.21182633 -21.29997346 -21.3549529  -21.4143523  -21.4691203
                    -21.53782513 -21.59646676 -21.63727847 -21.68928547 -21.71229674
                    -21.74681298 -21.79994631 -21.82187083 -21.82316103 -21.81809298
                    -21.84415125 -21.89031924 -21.91532362 -21.92233474 -21.90901291
                    -21.8865989  -21.93352953 -21.95220651 -21.96509542 -21.9748227
                    -22.01039857 -22.05452418 -22.06701185 -22.12118094 -22.17672083
                    -22.22156377 -22.28495559 -22.37424609 -22.43868249 -22.48726835
                    -22.52753463 -22.59043064 -22.63502323 -22.70609881 -22.76772107
                    -22.78999066 -22.89822463 -22.91642594 -22.91821851 -23.00260266
                    -23.02963462 -23.08855291 -23.14643934 -23.19314339 -23.26179212
                    -23.31983317 -23.37633646 -23.40913406 -23.46727623 -23.55360596
                    -23.62728585 -23.66199175 -23.65977512 -23.65637848 -23.6440235
                    -23.65515153 -23.75150412 -23.81719979 -23.86092942 -23.90322636
                    -23.93938644 -23.98060587 -24.0234856  -24.05883122 -24.10697327
                    -24.14803012 -24.20050986 -24.2461035  -24.25481232 -24.27006193
                    -24.27641934 -24.28806641 -24.30061744 -24.31734139 -24.29741394
                    -24.2394265  -24.28932435 -24.33346173 -24.33836285 -24.34911216
                    -24.35446391 -24.34999513 -24.34944438 -24.38088228 -24.40976198
                    -24.43689625 -24.48646791 -24.5440494  -24.60049291 -24.64022618
                    -24.67722542 -24.72940014 -24.74972424 -24.75169216 -24.82089282
                    -24.83281599 -24.97433036 -25.12130164 -25.18160631 -25.23465419
                    -25.27801117 -25.32730025 -25.39573122 -25.41811219 -25.41987955
                    -25.42251348 -25.42480562 -25.42578611 -25.4386075  -25.45809767
                    -25.4585384  -25.46898672 -25.50219363 -25.53438362 -25.5604614
                    -26.67812892])
            Standard deviation values for a Spectrogram Features in this dataset
                np.array([3.92355058 3.6502616  4.02182267 4.37552929 4.77734872 4.82466324
                    4.69020161 4.76369692 4.87766434 4.91796086 4.89056721 4.818071
                    4.76002897 4.69036622 4.58649688 4.48467655 4.3979366  4.30035466
                    4.20888081 4.17915974 4.12493582 4.11289304 4.11040908 4.09665318
                    4.07053862 4.03960701 4.00688875 3.86617317 3.85156668 3.97527887
                    4.00591881 4.0225738  4.01745245 4.03814252 4.02080265 4.00351084
                    4.01962589 4.01423651 4.03372691 4.01035597 3.98282137 4.00358362
                    4.00761433 4.00502736 4.01549732 4.01613916 4.00403415 3.99876117
                    3.9990366  3.99119954 3.97401411 3.97784257 3.97562441 3.98124039
                    3.9760159  3.9685926  3.98140347 3.98194729 4.00552542 4.00769776
                    3.95619025 3.98158054 4.00104463 3.99004035 3.97380202 3.98018278
                    3.97744922 3.94189778 3.93360745 3.92471709 3.9049791  3.88074674
                    3.86148902 3.8622598  3.87955714 3.88438454 3.88868102 3.8972226
                    3.8680993  3.82820872 3.78624699 3.80136514 3.78352835 3.77599828
                    3.85765369 3.88112438 3.90904495 3.90718194 3.92529367 3.93163349
                    3.89766358 3.86258285 3.84268727 3.84111327 3.84454305 3.83188163
                    3.83764017 3.85818298 3.88775697 3.86448318 3.84554978 3.8817106
                    3.8818073  3.89753403 3.92141535 3.92266064 3.91432112 3.90576029
                    3.88465077 3.86358514 3.83196789 3.842416   3.85064656 3.85023397
                    3.84901815 3.84828873 3.85354833 3.86728081 3.86216658 3.84256014
                    3.82077228 3.83352821 3.84533269 3.83578334 3.8178529  3.7936189
                    3.81024688 3.81379042 3.79121141 3.79064038 3.78858332 3.78676276
                    3.75337881 3.72550501 3.72131917 3.71580495 3.70462196 3.68096287
                    3.647642   3.70023362 3.71480128 3.76428922 3.81936224 3.8438355
                    3.86787062 3.87572879 3.88228615 3.86478947 3.8564306  3.86706149
                    3.87060719 3.88357981 3.89014793 3.88834421 3.87799328 3.87586469
                    3.8922332  3.89713341 3.90450808 3.90764932 3.92500374])
        """
        return (feature - self.feats_mean) / (self.feats_std + eps)

def shuffle_data(audio_paths, durations, texts):
    """ Shuffle the data (called after making a complete pass through 
        training or validation data during the training process)
    Params:
        audio_paths (list): Paths to audio clips
        durations (list): Durations of utterances for each audio clip
        texts (list): Sentences uttered in each audio clip
    """
    p = np.random.permutation(len(audio_paths))
    audio_paths = [audio_paths[i] for i in p] 
    durations = [durations[i] for i in p] 
    texts = [texts[i] for i in p]
    return audio_paths, durations, texts

def sort_data(audio_paths, durations, texts):
    """ Sort the data by duration 
    Params:
        audio_paths (list): Paths to audio clips
        durations (list): Durations of utterances for each audio clip
        texts (list): Sentences uttered in each audio clip
    """
    p = np.argsort(durations).tolist()
    audio_paths = [audio_paths[i] for i in p]
    durations = [durations[i] for i in p] 
    texts = [texts[i] for i in p]
    return audio_paths, durations, texts

def vis_train_features(index=0):
    """ Visualizing the data point in the training set at the supplied index
    """
    # obtain spectrogram
    audio_gen = AudioGenerator(spectrogram=True)
    audio_gen.load_train_data()
    vis_audio_path = audio_gen.train_audio_paths[index]
    vis_spectrogram_feature = audio_gen.normalize(audio_gen.featurize(vis_audio_path))
    # obtain mfcc
    audio_gen = AudioGenerator(spectrogram=False)
    audio_gen.load_train_data()
    vis_mfcc_feature = audio_gen.normalize(audio_gen.featurize(vis_audio_path))
    # obtain text label
    vis_text = audio_gen.train_texts[index]
    # obtain raw audio
    vis_raw_audio, _ = librosa.load(vis_audio_path)
    # print total number of training examples
    print('There are %d total training examples.' % len(audio_gen.train_audio_paths))
    # return labels for plotting
    return vis_text, vis_raw_audio, vis_mfcc_feature, vis_spectrogram_feature, vis_audio_path


def plot_raw_audio(vis_raw_audio):
    # plot the raw audio signal
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    steps = len(vis_raw_audio)
    ax.plot(np.linspace(1, steps, steps), vis_raw_audio)
    plt.title('Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()

def plot_mfcc_feature(vis_mfcc_feature):
    # plot the MFCC feature
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(vis_mfcc_feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_xticks(np.arange(0, 13, 2), minor=False)
    plt.show()

def plot_spectrogram_feature(vis_spectrogram_feature):
    # plot the normalized spectrogram
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(vis_spectrogram_feature, cmap=plt.cm.jet, aspect='auto')
    plt.title('Normalized Spectrogram')
    plt.ylabel('Time')
    plt.xlabel('Frequency')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

