# Imports

from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Activation, MaxPooling2D,  Dense, Dropout, Flatten, Conv2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model


class SatelliteModel():
    def __init__(self, X_train, Y_train, X_test, Y_test, patience, epochs, batch_size):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.patience = patience
        self.epochs = epochs
        self.batch_size = batch_size

    def build_model(self):
        model = Sequential()
        # Adds the first convulsion layer and follows up with max pooling
        model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(80, 80, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(80, 80, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        # Flattens the input into a 1D tensor
        model.add(Flatten())
        # Makes the input more readable for classification
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        # Final activation function
        model.add(Activation('sigmoid'))

        # Plot model
        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        # Summary
        print(model.summary())

        return model

    def compile_model(self):
        model = self.build_model()
        # Set learning_rate
        rmsprop = RMSprop(learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name="RMSprop")

        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
        return model

    def train_model(self):
        model = self.compile_model()
        # Early Stopping
        early_stopping = EarlyStopping(patience=self.patience)

        # Train the model
        model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size,
                                  validation_data=(self.X_test, self.Y_test),
                                  callbacks=[early_stopping])

        return model

    def make_prediction(self):
        model = self.train_model()

        # Predictions
        predictions = model.predict(self.X_test)

        return model, predictions
