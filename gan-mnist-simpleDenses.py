from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers

from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np



# Set the seed for reproducible result
np.random.seed(1000)

# Create a wall of generated MNIST images
def saveGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = G.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_%d.png' % epoch)



randomDim = 10

(x_train, _), (_,  _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(60000, 28*28)

G = Sequential()
G.add(Dense(256, input_dim=randomDim))
G.add(LeakyReLU(.2))
G.add(Dense(512))
G.add(LeakyReLU(.2))
G.add(Dense(1024))
G.add(LeakyReLU(.2))
G.add(Dense(784, activation='tanh'))

adam = Adam(learning_rate=0.0002, beta_1=0.5)

D = Sequential()
D.add(Dense(1024, input_dim=784))
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

dLosses = []
gLosses = []


def train(epochs=1, batchSize=128):
    batchCount = int(x_train.shape[0] / batchSize)
    print('Epoch:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0., 1, size=[batchSize, randomDim])
            imageBatch = x_train[np.random.randint(0, x_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            Gimgs = G.predict(noise)

            # print np.shape(imageBatch), np.shape(Gimgs)
            x = np.concatenate([imageBatch, Gimgs])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)

            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            D.trainable = True
            dloss = D.train_on_batch(x, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            D.trainable = False
            gloss = gan.train_on_batch(noise, yGen)
            
            # print(f"dLoss: {dloss}, gloss: {gloss}")

        dLosses.append(dloss) 
        gLosses.append(gloss)

        if e == 1 or e % 20 == 0:
            saveGeneratedImages(e)

    plotLoss(e)

train(epochs=200)



