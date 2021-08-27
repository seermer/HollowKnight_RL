import numpy as np
import tensorflow as tf

device = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(device, True)


class DQN:
    def __init__(self,
                 get_model,
                 buffer,
                 discount,
                 optimizer,
                 target_replace_step):
        self.buffer = buffer
        self.discount = discount
        self.target_replace_step = target_replace_step
        self.train_model = get_model()
        self.target_model = get_model()
        self.target_model.set_weights(self.train_model.get_weights())
        self.target_model.trainable = False
        self.step = 0
        self.train_model.compile(optimizer, loss=tf.keras.losses.mean_squared_error)


    @tf.function
    def train_step(self, s, a, r, s_, d):
        target_q = self.target_model(s_, training=False)
        d = 1. - tf.convert_to_tensor(d, dtype=target_q[0].dtype)
        r = tf.convert_to_tensor(r, dtype=target_q[0].dtype)
        for i in range(len(target_q)):
            target_q[i] = r + d * self.discount * tf.math.reduce_max(target_q[i], axis=1)
        with tf.GradientTape() as tape:
            train_out = self.train_model(s, training=True)
            losses = []
            for i, out in enumerate(train_out):
                q_train = tf.gather_nd(out, a[:, i:i + 1], batch_dims=1)
                losses.append(self.train_model.compiled_loss(target_q[i], q_train))
        grads = tape.gradient(losses, self.train_model.trainable_weights)
        self.train_model.optimizer.apply_gradients(zip(grads, self.train_model.trainable_weights))
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
        return tf.convert_to_tensor(
            [tf.argmax(arr[0]) for arr in self.train_model(tf.convert_to_tensor([s]), training=False)]
        )


if __name__ == '__main__':
    import model
    from functools import partial

    dqn = DQN(get_model=partial(model.get_model, in_shape=(256, 224, 3), frames=4, out_shape=[3, 3, 4], norm=True),
              buffer=None,
              discount=1.,
              optimizer=tf.keras.optimizers.Adam(clipnorm=10.),
              target_replace_step=1000)
    for _ in range(2):
        a_value = dqn.get_action(np.random.uniform(size=(4, 256, 224, 3)))
        # dqn.train_step(np.random.uniform(size=(2, 4, 256, 224, 3)), np.array([[1, 1, 3], [0, 2, 0]]), [.8, -.1],
        #                np.random.uniform(size=(2, 4, 256, 224, 3)), [False, True])
        print(a_value)
    import time

    t = time.time()
    for _ in range(50):
        dqn.get_action(np.random.uniform(size=(4, 256, 224, 3)))
        # dqn.train_step(np.random.uniform(size=(2, 4, 256, 224, 3)), np.array([[1, 1, 3], [0, 2, 0]]), [.8, -.1],
        #                np.random.uniform(size=(2, 4, 256, 224, 3)), [False, True])
    print((time.time() - t) / 50)
