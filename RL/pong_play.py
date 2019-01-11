import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.initializers import VarianceScaling
from keras.optimizers import RMSprop
import gym
import tensorflow as tf
import time


def huber_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


def build_model():
    model = Sequential()
    model.add(Conv2D(32, data_format="channels_first", input_shape=(1, 80, 80),
                     kernel_size=(8, 8), strides=4, padding="same",
                     activation="relu", kernel_initializer=VarianceScaling()))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=2, padding="same",
                     activation="relu", kernel_initializer=VarianceScaling()))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, padding="same",
                     activation="relu", kernel_initializer=VarianceScaling()))
    model.add(Flatten())
    model.add(Dense(512, activation="relu", kernel_initializer=VarianceScaling()))
    model.add(Dense(2, activation="linear", kernel_initializer=VarianceScaling()))
    model.compile(loss=huber_loss, optimizer=RMSprop(lr=0.00004))
    #model.load_weights("4stepv1.h5")
    model.load_weights("oct18.h5")
    return model


def epsilon_greedy(probs, epsilon):
    return np.random.randint(2) if np.random.random() < epsilon else np.argmax(probs)


def prepro(s):
    s = s[35:195]
    s = s[::2, ::2, 0]
    s[s == 144] = 0
    s[s == 109] = 0
    s[s != 0] = 1
    return s.astype(np.float32).reshape(-1, 1, 80, 80)


def overlay(states8080, dim_factor=0.5):
    dimmed_states = [state * dim_factor**index
                     for index, state in enumerate(reversed(states8080))]
    return np.max(np.array(dimmed_states), axis=0)


def play():
    env = gym.make("PongNoFrameskip-v4")
    model = build_model()
    while True:
        s = env.reset()
        for _ in range(np.random.randint(10)):
            s, _, _, _ = env.step(0)
        current = prepro(s)
        prev, prev2, prev3 = prepro(np.zeros_like(s)), prepro(np.zeros_like(s)), prepro(np.zeros_like(s))
        diff = overlay([prev3, prev2, prev, current])
        while True:
            a = epsilon_greedy(model.predict(diff.reshape(-1, 1, 80, 80)).flatten(), 0.025)
            for _ in range(4):
                s1, r, d, _ = env.step(a + 2)
                if d or r != 0:
                    break
            env.render()
            time.sleep(0.01)
            if d:
                break
            prev3 = prev2; prev2 = prev; prev = current; current = prepro(s1)
            diff = overlay([prev3, prev2, prev, current])


if __name__ == '__main__':
    play()
