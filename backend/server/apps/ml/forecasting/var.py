import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR

class VAR:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.train_df_reduced =  joblib.load(path_to_artifacts + "train_df_reduced.joblib")
        self.train_df_reduced_diff =  joblib.load(path_to_artifacts + "train_df_reduced_diff.joblib")
        self.minimum =  joblib.load(path_to_artifacts + "minimum.joblib")
        self.maximum =  joblib.load(path_to_artifacts + "maximum.joblib")
        self.product_list = joblib.load(path_to_artifacts + "product_list.joblib")
        self.customer_list = joblib.load(path_to_artifacts + "customer_list.joblib")
        self.pca = joblib.load(path_to_artifacts + "pca.joblib")
        self.var = joblib.load(path_to_artifacts + "var.joblib")

    def predict(self, number_of_days):
        return self.var.forecast(self.train_df_reduced_diff[-1:], steps=number_of_days)

    def postprocessing(self, prediction_reduced_diff):
        prediction_reduced = self.train_df_reduced[-1] + np.cumsum(prediction_reduced_diff, axis=0)
        prediction = self.pca.inverse_transform(prediction_reduced)
        prediction_denormalized = prediction * (self.maximum.to_numpy() - self.minimum.to_numpy()) + self.minimum.to_numpy()
        return prediction_denormalized
    
    def getexactprediction(self, predictions, item_Id, customer_Id):
        product_index = np.where(self.product_list == item_Id)[0][0]
        customer_index = np.where(self.customer_list == customer_Id)[0][0]
        index = product_index * (len(self.customer_list)) + customer_index
        return predictions[:, index]

    def compute_prediction(self, input_data):
        try:
            prediction = self.predict(input_data["number_of_days"])
            prediction = self.postprocessing(prediction)
            return self.getexactprediction(prediction, input_data["item_id"], input_data["customer_id"])
        except Exception as e:
            return {"status": "Error", "message": str(e)}