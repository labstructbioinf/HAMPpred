import os

from keras import Input, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ProgbarLogger, ReduceLROnPlateau
from keras.models import Model

from hamp_pred.utils.numpy_json import are_arrays_with_data


class BaseWrapper:
    def __init__(self, name, data_dir=None, use_case=None, version=None):
        self._name = name
        self.use_case = use_case
        self.version = version
        self.data_dir = data_dir
        self._model = None
        self.config = {}
        self.history = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def weights_path(self) -> str:
        return os.path.join(self.data_dir, f'{self.name}.h5')

    def train(self, X_train, y_train, X_test=None,
              y_test=None):
        if not os.path.exists(os.path.dirname(self.weights_path)):
            os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        self.history = self._model.fit(X_train, y_train,
                                       epochs=self.config.get('epochs', 60),
                                       verbose=True,
                                       validation_data=(X_test, y_test) if are_arrays_with_data((X_test, y_test)) else None,
                                       batch_size=64,
                                       callbacks=list(self.callbacks()))
        self._model.load_weights(self.weights_path)
        return self

    def callbacks(self):
        logger = ProgbarLogger(stateful_metrics=['mse'])
        earlyStopping = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(self.weights_path, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4,
                                           mode='min')
        return earlyStopping, mcp_save, reduce_lr_loss, logger

    def predict(self, X_data, **kwargs):
        self._model.load_weights(self.weights_path)
        return self._model.predict(X_data,
                                   batch_size=kwargs.get('batch_size', 1024),
                                   verbose=True)

    def compile(self, *args, **kwargs):
        self._model.compile(*args, **kwargs)
        return self

    def fit(self, *args, **kwargs):
        self._model.fit(*args, **kwargs)
        return self

    def test(self, *args, **kwargs):
        result = self._model.evaluate(*args, **kwargs)
        return result


class BaseLinear(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = layers.Dense(100, activation='relu')
        self.dense2 = layers.Dense(100, activation='relu')
        self.dropout1 = layers.Dropout(0.5)

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        if training:
            x = self.dropout1(x, training=training)
        return self.dense2(x)


class BaseLinearWrapper(BaseWrapper):
    def __init__(self, name='base_linear', config=None):
        config = config or {}
        super().__init__(name, config.get('data_dir'), config.get('task'))
        self.config = config
        self._model_schema = BaseLinear()
        self._model = None

    def _schema(self, inp):
        dense1 = layers.Dense(100, activation='relu')(inp)
        self.dense2 = layers.Dense(100, activation='relu')

    def build_many_inp_out(self, inp_shape, out_shape, n_inp=2, n_out=2):
        inps = []
        for inp in range(n_inp):
            inps.append(Input(shape=inp_shape, name=f"inp{inp}"))
        concate_input = layers.Concatenate()(inps)
        model = self._model_schema(concate_input)

    def build(self, inp_shape, out_shape):
        inp = Input(inp_shape)
        model = self._model_schema(inp)
        out = layers.TimeDistributed(layers.Dense(out_shape))(model)
        self._model = Model(inputs=inp, outputs=out)
        return self

    def train(self, X_train, y_train, X_test=None,
              y_test=None):
        return super().train(X_train, y_train, X_test=X_test,
                             y_test=y_test)


class BaseConvolutionalWrapper(BaseWrapper):
    def __init__(self, name='base_convolutional', config=None):
        config = config or {}
        super().__init__(name, config.get('data_dir'), config.get('task'), config.get('version'))
        self.config = config

    def _schema(self, inp, n_layers=3, filters=64, kernel_sizes=(7, 5, 3, 11, 14), norm=False, **kwargs):
        inp = layers.Masking(mask_value=0., input_shape=(inp.shape[1], inp.shape[2]))(inp)
        for i in range(n_layers):
            res = []
            for kr in kernel_sizes:
                res.append(layers.Conv1D(filters, kr, padding='same', activation='tanh')(inp))
            conc = layers.concatenate(res)
            inp = conc
            if norm:
                inp = layers.BatchNormalization()(inp)
        if kwargs.get('lstm'):
            for i in range(kwargs.get('lstm')):
                inp = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
        for i in range(kwargs.get('dense', 0)):
            inp = layers.Dense(200, activation='tanh')(inp)
        return layers.Dense(200, activation='tanh')(inp)

    def build(self, inp_shape, out_shape):
        inp = Input(inp_shape)
        schema = self._schema(inp, **self.config)
        out = layers.Dense(out_shape, activation=self.config.get('activation'))(schema)
        model = Model(inputs=inp, outputs=out)
        self._model = model
        return self

    def build_many_inp_out(self, inp_shape, out_shape, n_inp=2, n_out=2):
        inps = []
        for inp in range(n_inp):
            inps.append(Input(shape=inp_shape, name=f"inp{inp}"))
        concate_input = layers.Concatenate()(inps)
        model = self._schema(concate_input)
        outs = []
        for i in range(n_out):
            outs.append(layers.Dense(out_shape, activation=self.config.get('activation'))(model))
        encoder = Model(inputs=inps, outputs=outs, name=self.name)
        self._model = encoder
        return self

    def train(self, X_train, y_train, X_test=None,
              y_test=None):
        return super().train(X_train, y_train, X_test=X_test,
                             y_test=y_test)
