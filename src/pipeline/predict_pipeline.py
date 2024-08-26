import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(
        self,
        Time: float,
        Ipv: float,
        Vpv: float,
        Vdc: float,
        ia: float,
        ib: float,
        ic: float,
        va: float,
        vb: float,
        vc: float,
        Iabc: float,
        If: float,
        Vabc: float,
        Vf: float
    ):
        self.Time = Time
        self.Ipv = Ipv
        self.Vpv = Vpv
        self.Vdc = Vdc
        self.ia = ia
        self.ib = ib
        self.ic = ic
        self.va = va
        self.vb = vb
        self.vc = vc
        self.Iabc = Iabc
        self.If = If
        self.Vabc = Vabc
        self.Vf = Vf

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Time": [self.Time],
                "Ipv": [self.Ipv],
                "Vpv": [self.Vpv],
                "Vdc": [self.Vdc],
                "ia": [self.ia],
                "ib": [self.ib],
                "ic": [self.ic],
                "va": [self.va],
                "vb": [self.vb],
                "vc": [self.vc],
                "Iabc": [self.Iabc],
                "If": [self.If],
                "Vabc": [self.Vabc],
                "Vf": [self.Vf],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
