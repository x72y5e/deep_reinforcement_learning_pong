import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.initializers import VarianceScaling
from keras.optimizers import RMSprop
import gym
from collections import deque
from rl_utils import huber_loss, epsilon_greedy, prepro, overlay


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
    model.compile(loss=huber_loss, optimizer=RMSprop(lr=0.0002))
    return model


def initialize_model(model, graph):
    s = prepro(gym.make("PongNoFrameskip-v4").reset()).reshape(-1, 1, 80, 80)
    with graph.as_default():
        model.fit(s, np.array([0.5, 0.5]).reshape(-1, 2), epochs=1, verbose=0)


def train(model, target_model, replay_buffer, gamma=.99, batch_size=128):
    if len(replay_buffer) < 10000:
        print("insufficient training data.")
        return model, target_model

    X = np.zeros((batch_size, 1, 80, 80))
    y = np.zeros((batch_size, 2))
    i = 0

    batch = np.array(replay_buffer)[np.random.choice(range(len(replay_buffer)), batch_size), :]

    for (s, a, r, s1, d) in batch:
        if r != 0 or d:
            reward = r
        else:
            reward = r + gamma * np.amax(target_model.predict(s1.reshape(-1, 1, 80, 80)))
        target = model.predict(s.reshape(-1, 1, 80, 80)).reshape(2)
        target[a] = reward
        X[i] = s
        y[i] = target
        i += 1

    model.fit(X, y, epochs=1, verbose=0)
    return model, target_model


def play(epsilon=1., render=False):
    env = gym.make("PongNoFrameskip-v4")
    model = build_model()
    target_model = build_model()
    model.load_weights("oct18.h5")
    model.summary()

    replay_buffer = deque([], maxlen=75000)
    eps_decay = (1. - .02) / 25000
    rewards = deque([], maxlen=40)
    epochs = 0

    # outer loop
    while True:
        s = env.reset()
        initial = prepro(np.zeros_like(s))
        current = prepro(s)
        prev, prev2, prev3 = initial, initial, initial
        diff = overlay([prev3, prev2, prev, current])
        frame_limit = 2000

        # inner loop
        for frame in range(frame_limit):

            if epochs % 100 == 0:
                target_model.set_weights(model.get_weights())
                model.save_weights("oct18.h5")
            epochs += 1

            if frame % 4 == 0 and len(replay_buffer) > 10000:
                model, target_model = train(model, target_model, replay_buffer)
                epsilon = max(0.02, epsilon - eps_decay)
                print("epoch: {} - mean reward: {} - epsilon: {}".format(
                    epochs,
                    round(np.mean(rewards), 3),
                    round(epsilon, 3)))

            p = target_model.predict(diff.reshape(-1, 1, 80, 80))
            a = epsilon_greedy(p.flatten(), epsilon)
            agg_reward = 0
            for _ in range(4):
                s1, r, d, _ = env.step(a + 2)
                agg_reward += r
                if d or r != 0:
                    break
            if render:
                env.render()
            prev3 = prev2; prev2 = prev; prev = current; current = prepro(s1)
            next_diff = overlay([prev3, prev2, prev, current])
            replay_buffer.append((diff, a, agg_reward, next_diff, d))
            diff = next_diff
            if r != 0:
                rewards.append(r)
            if d or frame == frame_limit - 1:# or r != 0:
                print("game boundary (reward: {} - frames: {})".format(r, frame))
                break


if __name__ == '__main__':
    play(render=False, epsilon=.05)
