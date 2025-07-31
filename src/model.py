import os

print(f'''Keras backend: {os.environ['KERAS_BACKEND']}''')

from typing import override
import keras
from keras import ops
from keras.layers import Dense
from keras import KerasTensor
from numpy import pi


@keras.saving.register_keras_serializable()
class GaussianLayer(keras.Layer):
    def __init__(self, out_size: int=1, **kwargs):
        super().__init__(**kwargs)
        self.out_size: int = out_size
        self.layer_mu = Dense(out_size)
        self.layer_sigma = Dense(out_size, activation='elu')

    @override
    def call(self, inputs: KerasTensor) -> KerasTensor:
        mu: KerasTensor = self.layer_mu(inputs)

        sigma: KerasTensor = self.layer_sigma(inputs)
        sigma = ops.clip(sigma, 0.1, 50)
        sigma = sigma + 1.1

        result: KerasTensor = ops.hstack([mu, sigma])

        return result

    @override
    def get_config(self):
        config = super().get_config()
        config.update({'out_size': self.out_size})
        return config


@keras.saving.register_keras_serializable()
class GaussianLayerTwoInput(keras.Layer):
    def __init__(self, out_size: int=1, **kwargs):
        super().__init__(**kwargs)
        self.out_size: int = out_size
        self.layer_mu = Dense(out_size)
        self.layer_sigma = Dense(out_size, activation='elu')

    @override
    def call(self, mu_inputs: KerasTensor, sigma_inputs: KerasTensor) -> KerasTensor:
        mu: KerasTensor = self.layer_mu(mu_inputs)

        sigma: KerasTensor = self.layer_sigma(sigma_inputs)
        sigma = ops.clip(sigma, 0.1, 50)
        sigma = sigma + 1.1

        result: KerasTensor = ops.hstack([mu, sigma])

        return result

    @override
    def get_config(self):
        config = super().get_config()
        config.update({'out_size': self.out_size})
        return config


@keras.saving.register_keras_serializable()
class GaussianProbabalisticLoss(keras.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def call(self, y_true, y_pred):
        mu = y_pred[:, 0]
        sigma = y_pred[:, 1]

        # from tensorflow probability: https://github.com/tensorflow/probability/blob/303e844fd6e80202410932fd52570850a5957500/tensorflow_probability/python/distributions/normal.py#L182
        log_unnormalized = -0.5 * (y_true/sigma - mu/sigma)**2
        log_normalization = 0.5 * ops.log(2*pi) + ops.log(sigma)
        log_prob = log_unnormalized - log_normalization

        result = ops.mean(-log_prob)

        #if result < 0:
        #    print('log_unnormalized=', log_unnormalized)
        #    print('log_normalization=', log_normalization)
        #    print('log_prob=', log_prob)
        #    print(result)
        #    print(y_true)
        #    print(y_pred)
        #    raise RuntimeError('Loss went negative???')

        return result
        #return ops.abs(ops.mean(-log_prob))


@keras.saving.register_keras_serializable()
class MultiLoss(keras.Loss):
    def __init__(self, losses: list[keras.Loss], weights=None, **kwargs):
        super().__init__(**kwargs)
        self.losses = losses
        if weights is not None:
            if len(weights) == len(losses):
                self.weights = weights
            else:
                raise ValueError(f'Number of weights ({len(weights)}) must be the same as the number of losses ({len(losses)})')
        else:
            self.weights = ops.repeat(1, len(losses))

    @override
    def call(self, y_true, y_pred):
        loss = self.losses[0](y_true, y_pred) * self.weights[0]
        for loss_func, weight in zip(self.losses[1:], self.weights[1:]):
            loss = loss + loss_func(y_true, y_pred) * weight
        return loss

    @override
    def get_config(self):
        config = super().get_config()
        config.update({'losses': self.losses})
        config.update({'weights': self.weights})
        return config


@keras.saving.register_keras_serializable()
class LossOnGaussian(keras.Loss):
    def __init__(self, loss: keras.Loss, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss
    
    @override
    def call(self, y_true, y_pred):
        return self.loss.call(y_true, y_pred[:,0])


@keras.saving.register_keras_serializable()
class LambdaLoss(keras.Loss):
    def __init__(self, loss_func, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss_func
    
    @override
    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)


@keras.saving.register_keras_serializable()
class MetricOnGaussian(keras.Metric):
    def __init__(self, metric: keras.Metric, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric

    @override
    def update_state(self, y_true, y_pred, *args, **kwargs):
        return self.metric.update_state(y_true, y_pred[:,0], *args, **kwargs)

    @override
    def result(self):
        return self.metric.result()


@keras.saving.register_keras_serializable()
class VariationalSamplingLayer(keras.Layer):
    def __init__(self, out_size: int=1, seed: int=329, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        self.rng = keras.random.SeedGenerator(seed)

        self.out_size: int = out_size
        self.layer_mu = Dense(out_size)
        self.layer_log_sigma = Dense(out_size)

    @override
    def call(self, inputs: KerasTensor, training=False) -> KerasTensor:
        mu: KerasTensor = self.layer_mu(inputs)
        log_sigma: KerasTensor = ops.clip(self.layer_log_sigma(inputs), -5, 5)

        if training:
            kl_loss = -0.5 * ops.mean(
                log_sigma - ops.square(mu) - ops.exp(log_sigma) + 1
            )
            self.add_loss(kl_loss)
            epsilon = keras.random.normal(mu.shape, seed=self.rng)
            return mu + ops.exp(log_sigma/2) * epsilon
        else:
            return mu + ops.exp(log_sigma/2)

    @override
    def get_config(self):
        config = super().get_config()
        config.update({'out_size': self.out_size})
        config.update({'seed': self.seed})
        return config


def mk_ann(layer_sizes: list[int], activation_function: str='selu') -> keras.Model:
    sequential_layers = keras.Sequential(
        [Dense(out, activation_function) for out in layer_sizes]
    )
    sequential_layers.add(Dense(1))

    return sequential_layers


def mk_pnn(layer_sizes: list[int], activation_function: str='selu') -> keras.Model:
    sequential_layers = keras.Sequential(
        [keras.layers.Dropout(0.1),*[Dense(out, activation_function) for out in layer_sizes]]
    )
    sequential_layers.add(GaussianLayer())

    return sequential_layers


def mk_fancy(input_size: int, layer_sizes: list[int], activation_function: str='selu') -> keras.Model:
    inputs = keras.Input(shape=(input_size,))

    sequential_layers = keras.Sequential([
        keras.layers.Dropout(0.1),
        *[layer
          for size in layer_sizes
          for layer in [
            Dense(size, activation_function),
            keras.layers.ActivityRegularization(l1=1e-8, l2=1e-6)]
          ]
    ])(inputs)

    mu_embedding = keras.Sequential([
        VariationalSamplingLayer(3),
        keras.layers.ActivityRegularization(l2=1e-5),
        Dense(16, 'selu'),
    ])(sequential_layers)

    sigma_embedding = keras.Sequential([
        VariationalSamplingLayer(3),
        keras.layers.ActivityRegularization(l2=1e-5),
        Dense(16, 'selu'),
    ])(sequential_layers)

    probabilistic_layer = GaussianLayerTwoInput()

    probability_output = probabilistic_layer(mu_embedding, sigma_embedding)

    return keras.Model(inputs=inputs, outputs=probability_output)
