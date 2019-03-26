import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import numpy as np
import pickle
from keras.utils import to_categorical
import keras.preprocessing as prep
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout
import matplotlib.pyplot as plt
import matplotlib
import datetime
matplotlib.use("Agg")

tf.test.gpu_device_name()


config = tf.ConfigProto()

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 1
K.set_session(tf.Session(config=config))

x_train, y_train, x_test, y_test, vocab_size, max_length = pickle.load(
    open("data/keras-data.pickle", "rb")).values()

#n=1000
#x_train = x_train[:n]
#y_train = y_train[:n]

x_train = prep.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = prep.sequence.pad_sequences(x_test, maxlen=max_length)


def build_model():
    model = Sequential()
    model.add(Embedding(vocab_size, output_dim=256))
    model.add(LSTM(256))
    model.add(Dense(2))
    model.add(Activation("softmax"))

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    return model


epochs = 15
with tf.device('/GPU:0'):
    # Building model
    model = build_model()

    # Fitting model
    y_binary = to_categorical(y_train, num_classes=2)

    filepath = "epoch_models/model-{epoch:02d}.h5"
    callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)]
    train_history = model.fit(x_train, y_binary, batch_size=128, epochs=epochs, callbacks=callbacks)
    model.save(f"LSTM_ReviewClf_{datetime.datetime.now()}.h5")

    # Plotting training epochs
    print("[INFO] plotting epoch figure...")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), train_history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), train_history.history["acc"], label="train_acc")
    plt.title("Loss/Accuracy on Reviews")
    plt.xlabel("epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('epoch_fig')
