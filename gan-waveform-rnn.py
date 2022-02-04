from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, SimpleRNN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

from tqdm import tqdm
import os, time
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf



# Set the seed for reproducible result
np.random.seed(1000)

randomDim = 1000

# Saving generated wave samples
def saveGeneratedWaves(epoch, examples=10, dim=(10, 1), figsize=(10, 10)):
    trigger = np.random.normal(0, 1, size=[examples, randomDim])
    generatedWaves = G.predict(trigger)
    generatedWaves = generatedWaves.reshape(examples, blockSize)

    plt.figure(figsize=figsize)
    for i in range(generatedWaves.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.plot(generatedWaves[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('../data/output/images/gan_generated_wave_epoch_%d.png' % epoch)
    sf.write(
        file=f"..\data\output\waves\gan_generated_wave_epoch_{epoch}.wav",
        data=generatedWaves.reshape(examples*blockSize),
        samplerate=resamplerate)

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(lossesD, label='Discriminitive loss')
    plt.plot(lossesG, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('..\\data\\output\\images\\gan_loss_epoch_%d.png' % epoch)


def isValidDir(loc):
    return True if(os.path.isdir(loc)) else False

def load_wav_dir(dir_wav, duration=60):

    start_time = time.time()

    # listup files from given dir_wav
    if(isValidDir(dir_wav)):
        lst_file = os.listdir(dir_wav)
        lst_file = [s for s in lst_file if (".wav" in s) or (".WAV" in s)]
        n_files = len(lst_file)
        lst_file = [os.path.join(dir_wav, s) for s in lst_file]
        lst_sr = [librosa.get_samplerate(s) for s in lst_file]
        lst_size = [os.path.getsize(s) for s in lst_file]
        # lst_len = [a/b/4 for a, b in zip(lst_size, lst_sr)]
        dct_file = {k: [v, l] for k, v, l in zip(lst_file, lst_size, lst_sr)}
        # dct_file = {k: v for k, v in zip(lst_file, lst_sr)}
        lst_file = sorted(dct_file.items(), key=lambda x: x[1])
    elif(os.path.isfile(dir_wav)):
        file_wav = [dir_wav]

    # loading wav files and putting waveforms into waves
    wfs = np.empty((0, blockSize), dtype=float)
    # _duration = 0
    idx = 0
    _wfs = 0
    _offset = offset
    _duration = int(duration/n_files)
    while(_wfs < duration):

        if(idx == len(lst_file)):
            idx = 0
            _offset += _duration
            pass
        if (len(lst_file) == 0):
            break
        else:
            pass

        loc = lst_file[idx][0]
        # _sr = lst_file[idx][1][1]
        _sr = resamplerate

        try:
            print("\tLoading target:{%s}..." % loc)
            wf, _ = librosa.load(
                loc,
                sr=_sr,
                mono=False,
                offset=_offset,
                duration=_duration,
                res_type='kaiser_best')

            # If it's stereo signal
            if(np.array(wf.shape)[0] == 2):
                wf = (wf[0, :] + wf[1, :]) / 2
                pass
            else:
                pass
            
            wf = (wf - np.min(wf)) / (np.max(wf)-np.min(wf))
            wf = wf + 1e-16

            n_blocks = wf.shape[0]//blockSize
            wf = wf[:n_blocks*blockSize]

            _len = librosa.get_duration(wf, sr)
            if(_len < _duration):
                lst_file.pop(idx)
                pass
            else:
                idx += 1
                pass
            pass

            wf = wf.reshape(wf.shape[0]//blockSize, blockSize)
            wfs = np.vstack((wfs, wf))
            _wfs += _len

            if _len < _duration:
                print("\tEntire duration of this file is shorter than the target duration: {%}s" % _len/_sr)
            else:
                
        
        except Exception as e:
            print("\t", e, ": file, {}".format(loc))
            # lst_file.remove(lst_file[j])
            pass

        pass
    _n_frame = int(_sr/blockSize*duration)
    if wfs.shape[0] > _n_frame:
        wfs = wfs[1:_n_frame+1, :]
        pass
    else:
        pass

    print("total duration: %2.3f seconds" % (wfs.shape[0]*wfs.shape[1]/sr))

    end_time = time.time()
    print("took {0:.1f} seconds...\n".format(end_time-start_time))

    # return wfs_class, {"class_type":class_type, "len_total":len_total, "n_files":n_files, "power_wfs":np.power(wfs_in_dir, 2).mean()}
    return wfs


sr = 48000
resamplerate = sr
duration = 600
blockSize = int(resamplerate / 1)
blockSizes = [10, 50, 100, 200, 500, 1000]
unitLengths = [int(resamplerate/2/a) for a in blockSizes]
offset = 0

x_wav = load_wav_dir(dir_wav="..\\data\\input\\waves\\", duration=duration)


Ig = Input(shape=(1, resamplerate))
R1g = layers.SimpleRNN(units=unitLengths[0])(Ig)



G.add(Dense(512, input_dim=randomDim))
G.add(LeakyReLU(.2))
G.add(Dense(1024))
G.add(LeakyReLU(.2))
G.add(Dense(2048))
G.add(LeakyReLU(.2))
G.add(Dense(blockSize, activation='tanh'))

adam = Adam(learning_rate=.0001, beta_1=0.5)

D = Sequential()
D.add(Dense(1024, input_dim=blockSize))
D.add(LeakyReLU(.2))
D.add(Dropout(.3))
D.add(Dense(512))
D.add(LeakyReLU(.2))
D.add(Dropout(.3))
D.add(Dense(256))
D.add(LeakyReLU(.2))
D.add(Dropout(.3))
D.add(Dense(1, activation='sigmoid'))
D.compile(loss='binary_crossentropy', optimizer=adam)

D.trainable = False
ganInput = Input(shape=(randomDim,))
x = G(ganInput)
ganOutput = D(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

lossesD = []
lossesG = []


def train(epochs=1, batchSize=100):
    batchCount = int(x_wav.shape[0] / batchSize)
    print('Epoch:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):
            # Get a random set of input noise and images
            trigger = np.random.normal(0., 1, size=[batchSize, randomDim])
            waveBatch = x_wav[np.random.randint(0, x_wav.shape[0], size=batchSize)]

            # Generate fake MNIST images
            Gsample = G.predict(trigger)

            # print np.shape(waveBatch), np.shape(Gsample)
            x = np.concatenate([waveBatch, Gsample])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)

            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            D.trainable = True
            lossD = D.train_on_batch(x, yDis)

            # Train generator
            trigger = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            D.trainable = False
            lossG = gan.train_on_batch(trigger, yGen)
            
            # print(f"lossD: {lossD}, lossG: {lossG}")

        lossesD.append(lossD) 
        lossesG.append(lossG)

        if e == 1 or e % 20 == 0:
            saveGeneratedWaves(e)
            plotLoss(e)

    plotLoss(e)

train(epochs=10000)



