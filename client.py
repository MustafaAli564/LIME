import config
import tensorflow as tf
from tensorflow import keras
from lime.lime_tabular import LimeTabularExplainer


def create_model(input_dim):
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='elu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

class Client:
    def __init__(self,client_ID, data):
        self.client_ID = client_ID
        (self.x_train, self.y_train), (self.x_test, self.y_test), self.feature_names = data
        self.model = create_model(self.x_train.shape[1],)
        self.explainer = LimeTabularExplainer(
            training_data=self.x_train.values,  # Convert DataFrame to NumPy array
            training_labels=self.y_train,
            mode="classification",
            feature_names=self.feature_names,
            class_names=["normal", "DOS", "Probe", "R2L", "U2R"],
            discretize_continuous=True
        )


    def train(self):
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs = config.LOCAL_EPOCHS,
            batch_size = config.BATCH_SIZE,
            verbose = 1
        )
        return self.model.get_weights()
    
    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_stats(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        msg = f"Client_{self.client_ID} acc = {acc:.4f}; loss = {loss:.4f}"
        return msg

    def get_data_size(self):
        return self.x_train.shape[0]
    
    def get_input_dim(self):
        return self.x_train.shape[1]
    
    def getID(self):
        return self.client_ID
    
    def get_instance(self, i):
        print(self.x_test.shape)
        return self.x_test.iloc[i].values  # return NumPy array of the row


    def get_explanation(self, instance):
        explanation = self.explainer.explain_instance(
            instance,
            lambda x: self.model.predict(x),
            num_features=10
        )
        return explanation