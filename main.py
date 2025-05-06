from tensorflow import keras
import numpy as np
from client import Client
from server import Server
import config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import logging
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.read_csv("../Datasets/dataset.csv")

# Split features and labels
X = df.drop(columns=['label'])
y = df['label']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded, num_classes=23)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# X_scaled, y_one_hot = get_data()

def partitionData(client_id):
    # Split the data per client
    total_samples = len(X_scaled)
    samples_per_client = total_samples // config.NUM_CLIENTS
    start = client_id * samples_per_client
    end = (client_id + 1) * samples_per_client if client_id != config.NUM_CLIENTS - 1 else total_samples

    x_client = X_scaled[start:end]
    y_client = y_one_hot[start:end]

    # Train/test split for this client
    x_train, x_test, y_train, y_test = train_test_split(
        x_client, y_client, test_size=0.2, random_state=42
    )

    return (x_train, y_train), (x_test, y_test)
def preprocess_client1Data(path):
    try:
        logger.info(f"Loading data from {path}")
        df=pd.read_csv(path)
        
        logger.info("Creating dummy variables...")
        df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
        
        numeric_cols = [col for col in df.columns 
                       if col not in ['label', 'attack_category'] 
                       and not col.startswith(('protocol_type_', 'service_', 'flag_'))]
        
        logger.info(f"Scaling {len(numeric_cols)} numeric columns...")
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


       
        class_order = ['normal', 'DOS', 'Probe', 'R2L', 'U2R']
        encoder = OneHotEncoder(categories=[class_order], sparse_output=False, handle_unknown='ignore')
        Y = encoder.fit_transform(df[['attack_category']]).astype(np.float32)


        #encoder = OneHotEncoder(sparse_output=False)

        #Y = encoder.fit_transform(df[['attack_category']]).astype(np.float32)

        X = df.drop(['label', 'attack_category'], axis=1).astype(np.float32)

        logger.info(f"Final dataset shape - X: {X.shape}, Y: {Y.shape}")
        logger.info(f"Sample Y values: {Y[:5]}")
        #logger.info(f"Feature names: {df.columns.tolist()[-50:]}")  # Log last 50 features
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        return (x_train, y_train), (x_test,y_test), X.columns.tolist()
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}", exc_info=True)
        raise
def preprocess_client2Data(path):
    try:
        logger.info(f"Loading data from {path}")
        df=pd.read_csv(path)
        
        missing_columns_client2 = ['service_pm_dump', 'service_urh_i', 'flag_RSTOS0', 'service_tim_i', 'service_X11']

        # Add missing columns to client1_df if they are not present
        for col in missing_columns_client2:
            if col not in df.columns:
                df[col] = 0

        #drop the rows if their is any NAN in attack_catgory column
        df = df.dropna(subset=['attack_category'])
        
        # removing the duplicates if there is any
        df = df.drop_duplicates()
        logger.info(f"Data after cleaning: {df.shape}")
        
        #one-hot encoding for categorical columns
        logger.info("Creating dummy variables...")
        df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
        
        #extracting the numerical columns for further scaling
        numeric_cols = [col for col in df.columns 
                       if col not in ['label', 'attack_category'] 
                       and not col.startswith(('protocol_type_', 'service_', 'flag_'))]
        
        #scaling numerical columns using minmax
        logger.info(f"Scaling {len(numeric_cols)} numeric columns...")
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        
        class_order = ['normal', 'DOS', 'Probe', 'R2L', 'U2R']
        encoder = OneHotEncoder(categories=[class_order], sparse_output=False, handle_unknown='ignore')
        Y = encoder.fit_transform(df[['attack_category']]).astype(np.float32)
        X = df.drop(['label', 'attack_category'], axis=1).astype(np.float32)
        #logger.info(f"Feature names: {df.columns.tolist()[-50:]}") 
        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        return (x_train, y_train), (x_test,y_test), X.columns.tolist()
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}", exc_info=True)
        raise

def create_model(input_dim):
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(64, activation='gelu', input_shape=(X_scaled.shape[1],)),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(32, activation='gelu'),  # Hidden layer
    #     tf.keras.layers.Dropout(0.5), # Hidden layers
    #     tf.keras.layers.Dense(23, activation='softmax')  # 23
    # ])
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='elu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='elu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

client_list = []
client_list.append(Client(client_ID = 1, data=preprocess_client1Data("E:\Semesters Stuff\FYP\Datasets\clientnew1_data.csv")))
client_list.append(Client(client_ID = 2, data=preprocess_client2Data("E:\Semesters Stuff\FYP\Datasets\clientnew2_data.csv")))

global_model = create_model(client_list[0].get_input_dim())
server = Server(global_model)

for round in range(config.ROUNDS):
    print(f"--- Round {round+1} ---")
    global_weights = server.distribute()

    client_weights = []
    client_data_sizes = []
    if(round == config.ROUNDS - 1): stats = []
    for client in client_list:
        print(f"--- Client {client.getID()} ---")
        client.set_weights(global_weights)
        updated_weights = client.train()
        client_weights.append(updated_weights)
        client_data_sizes.append(client.get_data_size())
        if(round == config.ROUNDS - 1): stats.append(client.get_stats())


    server.aggregate(client_weights, client_data_sizes)
    
print(stats)
for i in range(10):
    instance =  client_list[0].get_instance(i)
    explanation = client_list[0].get_explanation(instance)
    explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.show()