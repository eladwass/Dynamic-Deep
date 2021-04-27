import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv1D, Flatten, Lambda, MaxPool1D, BatchNormalization, UpSampling1D, Concatenate, \
    Dropout
from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.layers import Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

import Enviroments.ExternalEnv.EcgClassification.ecg.network as network

acc_metric_first = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
acc_metric_second = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
acc_metric = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")


def slide_window(model, signal):
    lower_part = Lambda(lambda y: y[:, :5000, ])(signal)

    higher_part = model(Lambda(lambda y: y[:, 8960 - 5000:, ])(signal))
    higher_part = Lambda(lambda x: x[:, (10000 - 8960):])(higher_part)

    return Concatenate(axis=1)([model(lower_part), higher_part])


class PolicyPredictor_dense_ANDREW(Model):
    def __init__(self, **kwargs):
        super(PolicyPredictor_dense_ANDREW, self).__init__(**kwargs)

        self.dropout = Dropout(rate=0.2)

        self.dense_4 = Dense(1, activation=tf.keras.activations.relu)

    def call(self, inputs, training=False):

        x = Flatten()(inputs)


        x = self.dropout(x)

        x = self.dense_4(x)
        return x

class PolicyPredictor_dense_new(Model):
    def __init__(self, **kwargs):
        super(PolicyPredictor_dense_new, self).__init__(**kwargs)
        alpha = 0.2
        self.conv_embeddings_1 = Conv1D(filters=4, kernel_size=3, activation=tf.keras.layers.LeakyReLU(alpha),
                                        padding='same')
        self.conv_embeddings_2 = Conv1D(filters=8, kernel_size=3, activation=tf.keras.layers.LeakyReLU(alpha),
                                        padding='same')
        self.conv_embeddings_25 = Conv1D(filters=8, strides=2, kernel_size=3,
                                         activation=tf.keras.layers.LeakyReLU(alpha), padding='same')
        self.conv_embeddings_3 = Conv1D(filters=4, kernel_size=3, activation=tf.keras.layers.LeakyReLU(alpha),
                                        padding='same')
        self.conv_embeddings_35 = Conv1D(filters=8, kernel_size=3, activation=tf.keras.layers.LeakyReLU(alpha),
                                         padding='same')
        self.conv_embeddings_4 = Conv1D(filters=8, strides=2, kernel_size=3,
                                        activation=tf.keras.layers.LeakyReLU(alpha), padding='same')
        self.conv_embeddings_45 = Conv1D(filters=1, kernel_size=3, activation=tf.keras.layers.LeakyReLU(alpha),
                                         padding='same')
        #         self.dropout = Dropout(rate=0.2)
        #         self.dense = Dense(252,  activation = tf.keras.layers.LeakyReLU(0.1))
        #         self.dense_2 = Dense(252,  activation = tf.keras.layers.LeakyReLU(0.1))
        #         self.dense_3 = Dense(252,  activation = tf.keras.layers.LeakyReLU(0.1))
        #         self.dense_25 = Dense(252, activation = tf.keras.layers.LeakyReLU(0.1))
        self.dense_35 = Dense(64, activation=tf.keras.layers.LeakyReLU(alpha))
        self.dense_4 = Dense(1, activation=tf.keras.layers.LeakyReLU(alpha))

    def call(self, inputs, training=False):
        x = self.conv_embeddings_1(inputs)
        x = self.conv_embeddings_2(x)
        x = self.conv_embeddings_25(x)
        x = self.conv_embeddings_3(x)
        x = self.conv_embeddings_35(x)
        x = self.conv_embeddings_4(x)
        x = self.conv_embeddings_45(x)
        x = Flatten()(x)

        #         x = self.dense(x)

        #         x = self.dense_2(x)
        #         x = self.dense_25(x)
        #         x = self.dense_3(x)
        #         x = self.dropout(x)
        x = self.dense_35(x)
        x = self.dense_4(x)
        return x

class PolicyPredictor_dense(Model):
    def __init__(self, **kwargs):
        super(PolicyPredictor_dense, self).__init__(**kwargs)
        self.conv_embeddings_1 = Conv1D(filters=4, kernel_size=3, activation='swish', padding='same')
        self.conv_embeddings_2 = Conv1D(filters=8, kernel_size=3, activation='swish', padding='same')
        self.conv_embeddings_25 = Conv1D(filters=8, strides=2, kernel_size=3, activation='swish', padding='same')
        self.conv_embeddings_3 = Conv1D(filters=4, kernel_size=3, activation='swish', padding='same')
        self.conv_embeddings_35 = Conv1D(filters=8, kernel_size=3, activation='swish', padding='same')
        self.conv_embeddings_4 = Conv1D(filters=8, strides=2, kernel_size=3, activation='swish', padding='same')
        self.conv_embeddings_45 = Conv1D(filters=1, kernel_size=3, activation='swish', padding='same')
        self.dropout = Dropout(rate=0.2)
#         self.dense = Dense(31, activation=tf.keras.activations.relu)
#         self.dense_2 = Dense(128, activation=tf.keras.activations.swish)
#         self.dense_3 = Dense(64, activation=tf.keras.activations.swish)
        self.dense_35 = Dense(16, activation=tf.keras.activations.swish)
        self.dense_4 = Dense(1, activation=tf.keras.activations.swish)

    def call(self, inputs, training=False):
        x = self.conv_embeddings_1(inputs)
        x = self.conv_embeddings_2(x)
        x = self.conv_embeddings_25(x)
        x = self.conv_embeddings_3(x)
        x = self.conv_embeddings_35(x)
        x = self.conv_embeddings_4(x)
        x = self.conv_embeddings_45(x)
        x = Flatten()(x)

        # x = self.dense(x)
        x = self.dropout(x)
#         x = self.dense_2(x)
#         x = self.dense_3(x)
        x = self.dense_35(x)
        x = self.dense_4(x)
        return x



class CompressionAppNetworkSkip(Model):
    def __init__(self, compression_net, app_net, predictor, lw, task_type='RECONS', **kwargs):
        super(CompressionAppNetworkSkip, self).__init__(**kwargs)
        self.compression_net = compression_net
        self.app_net = app_net
        self.embeddings_flatten_first = []
        self.embeddings_flatten_second = []
        self.reconstrcuted_signal_full_first = []
        self.reconstrcuted_signal_full_second = []
        self.task_type = task_type

        self.policy_predictor_first = predictor[0]
        self.policy_predictor_second = predictor[1]
        self.dense_critic_1 = Dense(1, activation=tf.keras.activations.relu)
        # self.dense_critic_2 = Dense(1, activation=tf.keras.activations.relu)
        self.loss_weigts = lw

    def compile(self, optimizer, loss):
        super(CompressionAppNetworkSkip, self).compile()

    def call(self, inputs, training=False):
        app_model, model_ecg_classification = self.app_net
        self.embeddings_flatten_first = []
        self.embeddings_flatten_second = []
        self.reconstrcuted_signal_full_first = []
        self.reconstrcuted_signal_full_second = []
        for i in range(4):
            offset_input = Lambda(lambda y: y[:, i * 2000:(i + 1) * 2000, ])(inputs)
            offset_embeddings_32, offset_embeddings_64 = self.compression_net.layers[1](offset_input)

            offset_recons_32 = self.compression_net.layers[4](ZeroPadding1D(padding=(0, 32))(offset_embeddings_32))
            self.reconstrcuted_signal_full_first.append(offset_recons_32)
            offset_recons_64 = self.compression_net.layers[4](ZeroPadding1D(padding=(63, 0))(offset_embeddings_64))
            self.reconstrcuted_signal_full_second.append(offset_recons_64)

            offset_embeddings_32 = self.policy_predictor_first(offset_embeddings_32)
            self.embeddings_flatten_first.append(offset_embeddings_32)
            offset_embeddings_64 = self.policy_predictor_second(offset_embeddings_64)
            self.embeddings_flatten_second.append(offset_embeddings_64)

        x_reconstructed_first = Concatenate(axis=1)(self.reconstrcuted_signal_full_first)
        x_reconstructed_second = Concatenate(axis=1)(self.reconstrcuted_signal_full_second)
        x_skip = Lambda(lambda y: y[:, 8000:, ])(inputs)
        out_reconstrcuted_first = Concatenate(axis=1, name='out_recons_first')([x_reconstructed_first, x_skip])
        out_reconstrcuted_second = Concatenate(axis=1, name='out_recons_second')([x_reconstructed_second, x_skip])

        x_downstream_loss_first = Concatenate(axis=1)(self.embeddings_flatten_first)
        x_downstream_loss_second = Concatenate(axis=1)(self.embeddings_flatten_second)
        x_downstream_loss_pred_first = self.dense_critic_1(x_downstream_loss_first)
        x_downstream_loss_pred_second = self.dense_critic_1(x_downstream_loss_second)
        # x_reconstructed_loss_pred_first = self.dense_critic_2(x_downstream_loss_first)
        # x_reconstructed_loss_pred_second = self.dense_critic_2(x_downstream_loss_second)

        classification_first = model_ecg_classification(out_reconstrcuted_first)
        classification_second = model_ecg_classification(out_reconstrcuted_second)

        RR_first = slide_window(app_model, out_reconstrcuted_first)
        RR_second = slide_window(app_model, out_reconstrcuted_second)

        return [out_reconstrcuted_first, out_reconstrcuted_second,
                classification_first, classification_second,
                RR_first, RR_second,
                x_downstream_loss_pred_first, x_downstream_loss_pred_second]

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [acc_metric_first, acc_metric_second]

    def calculate_loss(self, annotation, app_result, state, reconstrcuted_signal, x_downstream_loss_pred,RR_loss,
                       downtask_type='RECONS'):

        task_1_label = annotation[2]
        task_2_label = annotation[4]

        cce = tf.keras.losses.CategoricalCrossentropy()
        app_loss = cce(task_1_label, app_result)

        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(state, reconstrcuted_signal))

        RR_loss = tf.reduce_mean(tf.keras.losses.Huber(reduction='none')(task_2_label, RR_loss))

        if downtask_type == 'RECONS':
            prediction_downstream_loss = tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(reconstruction_loss, x_downstream_loss_pred))
        elif downtask_type == 'APP_LOSS':
            prediction_downstream_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(app_loss, x_downstream_loss_pred))
        elif downtask_type == 'RR_LOSS':
            prediction_downstream_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(RR_loss, x_downstream_loss_pred))
        else: #Avarage loss of all tasks
            total_tasks_loss = self.loss_weigts['app_loss_weight'] * app_loss + \
                         self.loss_weigts['mse_loss_weight'] * reconstruction_loss + \
                         self.loss_weigts['RR_loss_weight']  * RR_loss

            prediction_downstream_loss = tf.reduce_mean(
                tf.keras.losses.mean_absolute_error(total_tasks_loss, x_downstream_loss_pred))

        # Since all weights are in the same scale (at least should be, there's no reason to scale:
        # total_prediction_loss = prediction_downstream_loss * 1#self.loss_weigts['prediction_downstream_loss_weight']

        total_loss = self.loss_weigts['app_loss_weight'] * app_loss + \
                    self.loss_weigts['mse_loss_weight'] * reconstruction_loss + \
                    self.loss_weigts['RR_loss_weight']  * RR_loss+\
                    self.loss_weigts['prediction_downstream_loss_weight'] * prediction_downstream_loss

        return total_loss, reconstruction_loss, app_loss, prediction_downstream_loss, RR_loss

    def train_step(self, data):
        state = data[0]
        annotation = data[1]
        #         (None, 35, 4) (None, 8960, 1)
        #         print(annotation.shape,state.shape)
        with tf.GradientTape() as tape:
            out_reconstrcuted_first, out_reconstrcuted_second, classification_first, classification_second,RR_first, RR_second, x_downstream_loss_pred_first, x_downstream_loss_pred_second = self(
                state, training=True)

            # Calculate loss of first compressor
            total_loss_first,reconstruction_loss, app_loss, prediction_downstream_loss, RR_loss = self.calculate_loss(
                annotation, classification_first, state, out_reconstrcuted_first, x_downstream_loss_pred_first,RR_first,
                self.task_type)

            acc_metric_first.update_state(annotation[2], classification_first)

            # Calculate loss of second compressor
            total_loss_second, reconstruction_loss_1, app_loss_1, prediction_downstream_loss_1, RR_loss_1 = self.calculate_loss(
                annotation, classification_second, state, out_reconstrcuted_second, x_downstream_loss_pred_second,RR_second,
                self.task_type)

            acc_metric_second.update_state(annotation[2], classification_second)

            total_loss_overall = total_loss_first + total_loss_second
        grads = tape.gradient(total_loss_overall, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        stats_dict = {
            "total_loss_1": total_loss_first,
            "mse_loss_1": reconstruction_loss,
            "app_loss_1": app_loss,
            "huber_loss_1": RR_loss,
            "prediction_down_loss_1": prediction_downstream_loss,
            'accuracy_1': acc_metric_first.result()
        }

        stats_dict_2 = {
            "total_loss_2": total_loss_second,
            "mse_loss_2": reconstruction_loss_1,
            "app_loss_2": app_loss_1,
            "huber_loss_2": RR_loss_1,
            "prediction_down_loss_2": prediction_downstream_loss_1,
            'accuracy_2': acc_metric_second.result()
        }
        return {**stats_dict, **stats_dict_2}

    def test_step(self, data, batch=False):
        # Unpack the data
        state = data[0]
        #if not batch:
        #    annotation = data[1][2]
        #else:
        annotation = data[1]

        out_reconstrcuted_first, out_reconstrcuted_second, classification_first, classification_second, RR_first, RR_second, x_downstream_loss_pred_first, x_downstream_loss_pred_second = self(
            state, training=False)

        # Calculate loss of first compressor
        total_loss_first, reconstruction_loss, app_loss, prediction_downstream_loss, RR_loss = self.calculate_loss(
            annotation, classification_first, state, out_reconstrcuted_first, x_downstream_loss_pred_first, RR_first,
            self.task_type)

        acc_metric_first.update_state(annotation[2], classification_first)

        # Calculate loss of second compressor
        total_loss_second, reconstruction_loss_1, app_loss_1, prediction_downstream_loss_1, RR_loss_1 = self.calculate_loss(
            annotation, classification_second, state, out_reconstrcuted_second, x_downstream_loss_pred_second,
            RR_second,
            self.task_type)

        acc_metric_second.update_state(annotation[2], classification_second)

        stats_dict = {
            "total_loss_1": total_loss_first,
            "mse_loss_1": reconstruction_loss,
            "app_loss_1": app_loss,
            "huber_loss_1": RR_loss,
            "prediction_down_loss_1": prediction_downstream_loss,
            'accuracy_1': acc_metric_first.result()
        }

        stats_dict_2 = {
            "total_loss_2": total_loss_second,
            "mse_loss_2": reconstruction_loss_1,
            "app_loss_2": app_loss_1,
            "huber_loss_2": RR_loss_1,
            "prediction_down_loss_2": prediction_downstream_loss_1,
            'accuracy_2': acc_metric_second.result()
        }

        total_loss = {'total_loss': total_loss_first + total_loss_second}

        return {**stats_dict, **stats_dict_2, **total_loss}


class CompressionAppNetwork(Model):
    def __init__(self, compression_net, app_net, predictor, lw, num_of_predictions=1, **kwargs):
        super(CompressionAppNetwork, self).__init__(**kwargs)
        self.compression_net = compression_net
        self.app_net = app_net
        self.policy_predictor_first = predictor[0]
        self.policy_predictor_second = predictor[1]
        self.dense_critic_1 = Dense(1,
                                    activation=tf.keras.activations.relu)  # TODO - which activation should I put here?
        self.dense_critic_2 = Dense(1,
                                    activation=tf.keras.activations.relu)  # TODO - which activation should I put here?
        self.loss_weigts = lw
        self.embeddings_flatten_first = []
        self.embeddings_flatten_second = []
        self.reconstrcuted_signal_full = []
        self.num_of_predictions = num_of_predictions

    def compile(self, optimizer, loss):
        super(CompressionAppNetwork, self).compile()

    def call(self, inputs, training=False):
        self.embeddings_flatten_first = []
        self.embeddings_flatten_second = []
        self.reconstrcuted_signal_full = []
        for i in range(4):
            offset_input = Lambda(lambda y: y[:, i * 2000:(i + 1) * 2000, ])(inputs)
            offset_embeddings = self.compression_net.layers[1](offset_input)
            offset_recons = self.compression_net.layers[2](offset_embeddings)
            self.reconstrcuted_signal_full.append(offset_recons)

            offset_embeddings_temp = self.policy_predictor_first(offset_embeddings)
            self.embeddings_flatten_first.append(offset_embeddings_temp)

            offset_embeddings_temp_2 = self.policy_predictor_second(offset_embeddings)
            self.embeddings_flatten_second.append(offset_embeddings_temp_2)

        x_reconstructed = Concatenate(axis=1)(self.reconstrcuted_signal_full)
        x_skip = Lambda(lambda y: y[:, 8000:, ])(inputs)
        out_reconstrcuted = Concatenate(axis=1, name='out_recons')([x_reconstructed, x_skip])

        x_recons_loss = Concatenate(axis=1)(self.embeddings_flatten_first)
        x_reconstructed_loss_pred = self.dense_critic_2(x_recons_loss)

        x_downstream_loss = Concatenate(axis=1)(self.embeddings_flatten_second)
        x_downstream_loss_pred = self.dense_critic_1(x_downstream_loss)

        classification = self.app_net(out_reconstrcuted)

        return [out_reconstrcuted, classification, x_reconstructed_loss_pred, x_downstream_loss_pred]

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [acc_metric]

    def train_step(self, data):
        state = data[0]
        annotation = data[1][2]
        #         (None, 35, 4) (None, 8960, 1)
        #         print(annotation.shape,state.shape)
        with tf.GradientTape() as tape:
            reconstrcuted_signal, app_result, x_reconstructed_loss_pred, x_downstream_loss_pred = self(state,
                                                                                                       training=True)

            # reconstructed_loss_prediction, downstream_loss_prediction = critic_value
            cce = tf.keras.losses.CategoricalCrossentropy()
            app_loss = cce(annotation, app_result)

            reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(state, reconstrcuted_signal))

            h = tf.keras.losses.mean_absolute_error
            prediction_downstream_loss = tf.reduce_mean(
                h(app_loss, x_downstream_loss_pred))
            prediction_recons_loss = tf.reduce_mean(
                h(reconstruction_loss, x_reconstructed_loss_pred))
            # total_prediction_loss = prediction_recons_loss*0 + prediction_downstream_loss

            total_loss = self.loss_weigts['app_loss_weight'] * app_loss + \
                         self.loss_weigts['mse_loss_weight'] * reconstruction_loss + \
                         self.loss_weigts['prediction_recons_loss_weight'] * prediction_recons_loss + \
                         self.loss_weigts['prediction_downstream_loss_weight'] * prediction_downstream_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        acc_metric.update_state(annotation, app_result)
        return {
            "loss": total_loss,
            "mse_loss": reconstruction_loss,
            "app_loss": app_loss,
            "prediction_recons_loss": prediction_recons_loss,
            "prediction_downstream_loss": prediction_downstream_loss,
            'accuracy': acc_metric.result()
        }

    def test_step(self, data, batch=False):
        # Unpack the data
        state = data[0]
        # if not batch:
        annotation = data[1][2]
        # else:
        #     annotation = data[2]

        reconstrcuted_signal, app_result, x_reconstructed_loss_pred, x_downstream_loss_pred = self(state,
                                                                                                   training=False)
        # reconstructed_loss_prediction, downstream_loss_prediction = critic_value
        cce = tf.keras.losses.CategoricalCrossentropy()
        app_loss = cce(annotation, app_result)

        reconstruction_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(state, reconstrcuted_signal))

        h = tf.keras.losses.mean_absolute_error
        prediction_downstream_loss = tf.reduce_mean(
            h(app_loss, x_downstream_loss_pred))
        prediction_recons_loss = tf.reduce_mean(
            h(reconstruction_loss, x_reconstructed_loss_pred))

        # total_prediction_loss = prediction_recons_loss * 0 + prediction_downstream_loss
        total_loss = self.loss_weigts['app_loss_weight'] * app_loss + \
                     self.loss_weigts['mse_loss_weight'] * reconstruction_loss + \
                     self.loss_weigts['prediction_recons_loss_weight'] * prediction_recons_loss + \
                     self.loss_weigts['prediction_downstream_loss_weight'] * prediction_downstream_loss

        acc_metric.update_state(annotation, app_result)

        return {
            "loss": total_loss,
            "mse_loss": reconstruction_loss,
            "app_loss": app_loss,
            "prediction_recons_loss": prediction_recons_loss,
            "prediction_downstream_loss": prediction_downstream_loss,
            'accuracy': acc_metric.result()
        }


default_lw = {
    'app_loss_weight': 0.2,
    'mse_loss_weight': 0.75,
    'RR_loss_weight' : 0.2,
    'prediction_recons_loss_weight': 0.05,
    'prediction_downstream_loss_weight': 0.05
}


def load_predictive_compressor(compressor, downstream_task, policy_predictor, weight_path, skip=False, lw=default_lw,
                               task_type='RECONS'):

    if skip:
        adaptive_model = CompressionAppNetworkSkip(compressor, downstream_task, policy_predictor, lw, task_type)
    else:
        adaptive_model = CompressionAppNetwork(compressor, downstream_task, policy_predictor, lw)

    opt = Adam(
        lr=0.001,
        decay=1e-5)

    adaptive_model.compile(opt, tf.losses.categorical_crossentropy)
    # adaptive_model_32.evaluate(x_test, y_test)
    if weight_path != '':
        adaptive_model.load_weights(weight_path).expect_partial()
    return adaptive_model
