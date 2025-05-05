import numpy as np

class Server:
    def __init__(self, model):
        self.global_model = model
    
    def aggregate(self, client_weights, client_data_sizes):
        # new_weights = list()
        # for weights_list_tuple in zip(*client_weights):
        #     new_weights.append(
        #         np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
        #     )
        # self.global_model.set_weights(new_weights)


        total_samples = sum(client_data_sizes)
        weighted_avg = [
            sum(client_weight[i] * (size / total_samples) for client_weight, size in zip(client_weights, client_data_sizes))
            for i in range(len(client_weights[0]))
        ]
        self.global_model.set_weights(weighted_avg)
        
    def distribute(self):
        return self.global_model.get_weights()