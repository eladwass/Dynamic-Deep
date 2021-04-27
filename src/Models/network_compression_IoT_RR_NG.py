from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv1D, Flatten, Lambda, MaxPool1D, BatchNormalization, UpSampling1D, Concatenate, \
    Dropout
from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.layers import Reshape, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import Enviroments.ExternalEnv.EcgClassification.ecg.network as network
from Enviroments.ExternalEnv.RPNet.network_keras import IncUNet


class Encoder_32_64_iot(Model):

    def __init__(self, norm_output, **kwargs):
        super(Encoder_32_64_iot, self).__init__(**kwargs)

        self.conv1 = Conv1D(filters=8, kernel_size=3, activation='swish', padding='same')
        self.conv12 = Conv1D(filters=16, kernel_size=3, strides=2, activation='swish', padding='same')
        self.conv15 = Conv1D(filters=8, kernel_size=1, activation='swish', padding='same')  # [1000 X 8]

        self.conv2 = Conv1D(filters=32, kernel_size=5, activation='swish', padding='same')
        self.bn1 = BatchNormalization()
        self.conv22 = Conv1D(filters=64, kernel_size=5, strides=2, activation='swish', padding='same')
        self.conv25 = Conv1D(filters=32, kernel_size=1, activation='swish', padding='same')  # [500X32]
        # 5
        self.conv3 = Conv1D(filters=16, kernel_size=3, activation='swish', padding='same', name='64_3')
        self.conv32 = Conv1D(filters=32, kernel_size=3, strides=2, activation='swish', padding='same', name='16_3')
        self.conv35 = Conv1D(filters=16, kernel_size=1, activation='swish', padding='same', name='16_3')  # [250 X 16]
        self.bn2 = BatchNormalization()

        self.conv4 = Conv1D(filters=32, kernel_size=11, activation='swish', padding='same', name='64_11')

        # 10
        self.conv5 = Conv1D(filters=64, kernel_size=7, strides=2, activation='swish', padding='same', name='128_13')
        self.conv55 = Conv1D(filters=64, kernel_size=1, activation='swish', padding='same', name='128_13')

        self.conv8 = Conv1D(filters=32, kernel_size=7, activation='swish', padding='same')

        self.conv9 = Conv1D(filters=64, kernel_size=5, strides=2, activation='swish', padding='same')
        self.conv95 = Conv1D(filters=32, kernel_size=1, activation='swish', padding='same')
        self.conv10 = Conv1D(filters=1, kernel_size=7, activation='swish', padding='same',
                             name='AE_encoder_output32CG_compressed')

        self.conv11 = Conv1D(filters=64, kernel_size=7, strides=2, activation='swish', padding='same',
                             name='AE_encoder_output64CG_compressed')
        self.conv13 = Conv1D(filters=1, kernel_size=3, activation='swish', padding='same', name='out_64')

    def call(self, inputs):
        # 15
        x = self.conv1(inputs)
        x = self.conv12(x)
        x = self.conv15(x)
        #         x = self.downsamp1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv22(x)
        x = self.conv25(x)

        # 5
        #         x = self.downsamp2(x)
        x = self.conv3(x)
        x = self.conv32(x)
        x = self.conv35(x)
        x = self.bn2(x)
        #         x = self.downsamp3(x)
        x = self.conv4(x)
        # 10
        x = self.conv5(x)
        x = self.conv55(x)
        #         x = self.conv6(x)
        #         x = self.downsamp4(x)
        #         x = self.conv7(x)
        x = self.conv8(x)  # [124X32]
        #         x = self.downsamp5(x)
        x = self.conv9(x)
        x = self.conv95(x)
        AE_encoder_output_32CG = self.conv10(x)  # [62X1]
        #         x = self.downsamp6(x)
        AE_encoder_output_64CG = self.conv11(x)
        AE_encoder_output_64CG = self.conv13(AE_encoder_output_64CG)
        return AE_encoder_output_32CG, AE_encoder_output_64CG


class Decoder_shared_iot(Model):

    def __init__(self, norm_output, **kwargs):
        super(Decoder_shared_iot, self).__init__(**kwargs)
        self.norm_output = norm_output

        self.upsam1 = UpSampling1D(size=2)
        self.zeroPadding = ZeroPadding1D(padding=(1, 1))
        self.conv3 = Conv1D(filters=64, kernel_size=7, activation='swish', padding='same')
        self.conv4 = Conv1D(filters=128, kernel_size=7, activation='swish', padding='same')
        # 20
        self.upsam2 = UpSampling1D(size=2)
        self.conv5 = Conv1D(filters=16, kernel_size=3, activation='swish', padding='same')
        self.conv6 = Conv1D(filters=32, kernel_size=5, activation='swish', padding='same')
        self.upsam3 = UpSampling1D(size=2)
        self.conv7 = Conv1D(filters=32, kernel_size=3, activation='swish', padding='same')
        # 25
        #         self.upsam4 = UpSampling1D(size=2)
        self.conv8 = Conv1D(filters=8, kernel_size=3, activation='swish', padding='same')
        self.conv9 = Conv1D(filters=2, kernel_size=3, activation='swish', padding='same')
        self.flatten = Flatten()

        self.Dense = Dense(2000, name='outputs_decoder')

    def call(self, inputs, training=False):
        y = self.upsam1(inputs)
        y = self.zeroPadding(y)
        # 15
        y = self.conv3(y)  # []
        y = self.conv4(y)
        # 20
        y = self.upsam2(y)
        y = self.conv5(y)
        y = self.conv6(y)
        y = self.upsam3(y)
        y = self.conv7(y)
        #         y = self.upsam4(y)
        # 25
        y = self.conv8(y)
        y = self.conv9(y)

        y = self.flatten(y)
        y = self.Dense(y)
        outputs_decoder = Reshape((2000, 1))(y)

        return outputs_decoder


def decoder_skip(iot=False):
    encoded_shape = (63 + 32, 1)
    inputs_combined = Input(shape=encoded_shape, name='encoder_input_combined')

    encoded32_input = Lambda(lambda y: y[:, 32:63 + 32, ])(inputs_combined)
    encoded64_input = Lambda(lambda y: y[:, 0:32, ])(inputs_combined)

    AE_encoder_output_32CG = Conv1D(filters=8, kernel_size=3, activation='swish', padding='same')(encoded32_input)
    AE_encoder_output_32CG = Conv1D(filters=16, kernel_size=3, activation='swish', padding='same')(
        AE_encoder_output_32CG)
    AE_encoder_output_32CG_32filters = UpSampling1D(size=2)(AE_encoder_output_32CG)  # [118X16]
    AE_encoder_output_32CG_32filters = Conv1D(filters=16, kernel_size=3, activation='swish', padding='same')(
        AE_encoder_output_32CG_32filters)  # [63X32]
    AE_encoder_output_32CG_32filters = Conv1D(filters=32, kernel_size=3, activation='swish', padding='valid')(
        AE_encoder_output_32CG_32filters)  # [124X32]

    encoded64_input = Conv1D(filters=16, kernel_size=7, activation='swish', padding='same')(encoded64_input)
    encoded64_input = Conv1D(filters=32, kernel_size=3, activation='swish', padding='same')(encoded64_input)
    encoded64_input = UpSampling1D(size=2)(encoded64_input)  # [64X32]
    encoded64_input = Conv1D(filters=32, kernel_size=3, activation='swish', padding='valid')(encoded64_input)
    encoded64_output = UpSampling1D(size=2)(encoded64_input)  # [124X32]

    if iot:
        decoder_shared = Decoder_shared_iot(True, name='decoder_32_64_shared_iot')
    else:
        pass  # decoder_shared = Decoder_16_32_shared(True, name='decoder_32_64_shared')
    decoded_shared_32 = decoder_shared(AE_encoder_output_32CG_32filters)
    decoded_shared_64 = decoder_shared(encoded64_output)

    Decoder_16_32_skip = Model(inputs_combined, [decoded_shared_32 + decoded_shared_64], name='decoder')
    return Decoder_16_32_skip


def u_net_32_64(path_weight, iot=True):
    '''
    relevent iot_weight: 'Models/Progressive/U_NET_32-64-IoT.hdf5' - total PRD - 13.3=5.92+7.41 or 0.5 + 0.7 10^-3 for MSE
    :param path_weight:
    :return:
    '''
    input_shape = (2000, 1)
    inputs = Input(shape=input_shape, name='encoder_input_combined')
    # 1
    encoder_32_output, encoder_64_output = Encoder_32_64_iot(True, name='adaptive_encoder')(inputs)
    Decoder_32_64_skip = decoder_skip(iot)
    recons_32CG = Decoder_32_64_skip(ZeroPadding1D(padding=(0, 32))(encoder_32_output))
    recons_64CG = Decoder_32_64_skip(ZeroPadding1D(padding=(63, 0))(encoder_64_output))

    # recons_16CG = Decoder_16_32(Concatenate(axis=1)([np.zeros([1,62,1]), encoder_16_output  ]))

    model_u_unet = Model(inputs, [recons_32CG, recons_64CG], name='U-NET_32_64_iot')

    if path_weight != "":
        print("loading compression net weights")
        model_u_unet.load_weights(path_weight)
    return model_u_unet


class CompressionAppNetwork(Model):
    def __init__(self, compression_net, app_net, **kwargs):
        super(CompressionAppNetwork, self).__init__(**kwargs)
        self.compression_net = compression_net
        self.app_net = app_net
        self.input_split = []

    def call(self, inputs, training=False):
        self.input_split = []
        for i in range(4):
            offset_input = Lambda(lambda y: y[:, i * 2000:(i + 1) * 2000, ])(inputs)
            offset_input = self.compression_net(offset_input)
            self.input_split.append(offset_input)

        x_reconstructed = Concatenate(axis=1)(self.input_split)
        x_skip = Lambda(lambda y: y[:, 8000:, ])(inputs)

        out_reconstrcuted = Concatenate(axis=1, name='out_recons')([x_reconstructed, x_skip])
        classification = self.app_net(out_reconstrcuted)

        return [out_reconstrcuted, classification]


def slide_window(model, signal):
    lower_part = Lambda(lambda y: y[:, :5000, ])(signal)

    higher_part = model(Lambda(lambda y: y[:, 8960 - 5000:, ])(signal))
    higher_part = Lambda(lambda x: x[:, (10000 - 8960):])(higher_part)

    return Concatenate(axis=1)([model(lower_part), higher_part])


class CompressionAppNetworkSkip(Model):
    '''
    /home/welad1992/thesis/Data/tensorsummery_unet_iot run #462
    sliding_ae = network_compression_IoT.init_compress_app_network(network_compression_IoT.u_net_32_64,'Models/Progressive/U_NET_32-64-IoT_V3.hdf5',params_config, compile=True, u_net=True)
    '''

    def __init__(self, compression_net, app_net, **kwargs):
        super(CompressionAppNetworkSkip, self).__init__(**kwargs)
        self.compression_net = compression_net
        self.app_net = app_net
        self.input_split_one = []
        self.input_split_second = []

    def call(self, inputs, training=False):
        app_model, model_ecg_classification = self.app_net
        self.input_split_one = []
        self.input_split_second = []

        for i in range(4):
            offset_input = Lambda(lambda y: y[:, i * 2000:(i + 1) * 2000, ])(inputs)
            offset_input = self.compression_net(offset_input)
            self.input_split_one.append(offset_input[0])
            self.input_split_second.append(offset_input[1])

        x_reconstructed_first = Concatenate(axis=1)(self.input_split_one)
        x_reconstructed_second = Concatenate(axis=1)(self.input_split_second)
        x_skip = Lambda(lambda y: y[:, 8000:, ])(inputs)

        out_reconstrcuted_first = Concatenate(axis=1, name='out_recons_low')([x_reconstructed_first, x_skip])
        out_reconstrcuted_second = Concatenate(axis=1, name='out_recons_high')([x_reconstructed_second, x_skip])

        RR_first = slide_window(app_model, out_reconstrcuted_first)
        RR_second = slide_window(app_model, out_reconstrcuted_second)

        classification_first = model_ecg_classification(out_reconstrcuted_first)
        classification_second = model_ecg_classification(out_reconstrcuted_second)

        return [out_reconstrcuted_first, out_reconstrcuted_second, classification_first, classification_second,
                RR_first, RR_second]


def compression_app_network(compression_net, app_net, u_net):
    if u_net:
        can = CompressionAppNetworkSkip(compression_net, app_net)
    else:
        can = CompressionAppNetwork(compression_net, app_net)
    input_shape = (8960, 1)
    compression_application_input = Input(shape=input_shape, name='compression_application_input')
    test = can(compression_application_input)

    return can


def init_compress_app_network(compression_net_builder, compression_weight_net_path, params_config, compile=False,
                              u_net=False):
    compression_net = compression_net_builder(compression_weight_net_path)

    app_model = IncUNet()
    inputs_signal = Input(shape=(5000, 1), name='input-RR')
    app_model(inputs_signal)
    app_model.load_weights('Models/RR/RR_app.hdf5')
    # Freeze application network layers
    for i in enumerate(app_model.layers):
        app_model.layers[i[0]].trainable = False

    model_ecg_classification = network.build_network(**params_config)
    # Freeze application network layers
    for i in enumerate(model_ecg_classification.layers):
        model_ecg_classification.layers[i[0]].trainable = False

    cae = compression_app_network(compression_net, [app_model, model_ecg_classification], u_net)

    if compile:
        # compile the new model:
        optimizer = Adam(
            lr=params_config["learning_rate"],
            clipnorm=params_config.get("clipnorm", 1), decay=1e-5)
        if u_net:
            losses = ["mse", "mse", "categorical_crossentropy", "categorical_crossentropy", "Huber", "Huber"]
            metrices = [[], [], ['categorical_accuracy'], ['categorical_accuracy'], [], []]

        else:
            losses = {
                cae.output_names[0]: "mse",
                cae.output_names[1]: "Huber"}
            metrices = {}

        cae.compile(optimizer=optimizer, loss=losses, metrics=metrices, loss_weights=params_config['loss_weights'])

    return cae

# params_config = {'batch_size': 32,
#  'conv_activation': 'relu',
#  'conv_dropout': 0.2,
#  'conv_filter_length': 16,
#  'conv_increase_channels_at': 4,
#  'conv_init': 'he_normal',
#  'conv_num_filters_start': 32,
#  'conv_num_skip': 2,
#  'conv_subsample_lengths': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
#  'dev': 'Enviroments/ExternalEnv/ecg-classification-AndrewNG/examples/cinc17/dev.json',
#  'generator': True,
#  'input_shape': [None, 1],
#  'learning_rate': 0.001,
#  'loss_weights': [0.5, 0.5],
#  'model_weights_path': '../src/Enviroments/ExternalEnv/ecg-classification-AndrewNG/saved/cinc17/Best_Weight/0.404-0.868-014-0.266-0.906.hdf5',
#  'num_categories': 4,
#  'save_dir': 'saved',
#  'train': 'Enviroments/ExternalEnv/ecg-classification-AndrewNG/examples/cinc17/train.json'}
#
# sliding_ae = init_compress_app_network(u_net_16_32_iot,'../Models/Progressive/U_NET_32-16_iot-[3.71][6.11].hdf5',params_config, compile=True, u_net=True)
# # pass
# #
