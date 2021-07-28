import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=D:/cuda/dev"
import numpy as np
import tensorflow as tf

tf.config.optimizer.set_jit("autoclustering")


class DQN:
    def __init__(self,
                 get_model,
                 buffer,
                 action_index,
                 discount,
                 optimizer,
                 target_replace_step):
        self.buffer = buffer
        self.discount = discount
        self.target_replace_step = target_replace_step
        self.train_model = get_model()

        self.target_model = get_model()
        self.step = 0
        self.loss_fn = tf.keras.losses.mean_squared_error
        self.train_model.compile(optimizer, loss=self.loss_fn)
        self.target_model.set_weights(self.train_model.get_weights())
        self.optimizer = optimizer
        self.action_index = action_index

    @tf.function
    def train_step(self, s, a, r, s_, d):
        target_q = self.target_model(s_, training=False)
        labels = []
        for i in range(len(target_q)):
            label = []
            for j in range(len(d)):
                if d[j]:
                    label.append(r[j])
                else:
                    label.append(r[j] + self.discount * tf.math.reduce_max(target_q[i][j]))
            label = tf.convert_to_tensor(label, dtype=target_q[i].dtype)
            labels.append(label)
        with tf.GradientTape() as tape:
            train_out = self.train_model(s, training=True)
            losses = []
            for i in range(len(train_out)):
                q_train = tf.gather(train_out[i], a[i], axis=1, batch_dims=-1)
                losses.append(self.loss_fn(labels[i], q_train))
        grads = tape.gradient(losses, self.train_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.train_model.trainable_weights))
        return losses

    def learn(self, batch_size=4):
        s, a, r, s_, d = self.buffer.sample(batch_size)
        losses = self.train_step(s, a, r, s_, d)
        self.step += 1
        if self.step % self.target_replace_step == 1:
            self.target_model.set_weights(self.train_model.get_weights())
            print("target model weights replaced")
        return losses

    @tf.function
    def get_action(self, s):
        return [tf.argmax(arr[0]) for arr in self.train_model(tf.convert_to_tensor([s]), training=False)]


if __name__ == '__main__':
    import model
    from functools import partial

    dqn = DQN(get_model=partial(model.get_model, in_shape=(256, 256, 3), frames=4, out_shape=[3, 3, 4], reduced=True),
              buffer=None,
              action_index=(2, 4),
              discount=.99,
              optimizer=tf.keras.optimizers.Adam(),
              target_replace_step=1000)
    # for _ in range(5):
    #     dqn.train_step(np.random.uniform(size=(3, 4, 256, 256, 3)), [[0, 2, 1], [2, 2, 1], [3, 0, 0]], [.5, -.8, .9],
    #                    np.random.uniform(size=(3, 4, 256, 256, 3)), [False, False, False])

    for _ in range(2):
        dqn.train_step(np.random.uniform(size=(1, 4, 256, 256, 3)), [[0], [1], [3]], [-.2],
                       np.random.uniform(size=(1, 4, 256, 256, 3)), [False])
    for _ in range(2):
        dqn.get_action(np.random.uniform(size=(4, 256, 256, 3)))
    import time

    # t = time.time()
    # for _ in range(30):
    #     dqn.train_step(np.random.uniform(size=(1, 4, 256, 256, 3)), [[0], [1], [3]], [-.2],
    #                    np.random.uniform(size=(1, 4, 256, 256, 3)), [False])
    # print((time.time() - t) / 30)
    # t = time.time()
    # for _ in range(30):
    #     dqn.train_step(np.random.uniform(size=(6, 4, 256, 256, 3)), np.random.randint(low=0, high=3, size=(3, 6)),
    #                    np.random.random(size=(6,)).astype(np.float32),
    #                    np.random.uniform(size=(6, 4, 256, 256, 3)), [False for _ in range(6)])
    # print((time.time() - t) / 30)

    # print(loss_out)
    t = time.time()
    for _ in range(50):
        dqn.get_action(np.random.uniform(size=(4, 256, 256, 3)))
    print((time.time() - t) / 50)
    # print(dqn.get_action(np.random.uniform(size=(4, 224, 224, 3))))
