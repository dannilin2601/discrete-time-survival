from ml_models_pipeline import run_coxph, run_nnet
import numpy as np
import pandas as pd
import tensorflow as tf

import random
import os

def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

df = pd.read_csv("data/df_normal.csv")

print("Neural Network Model Results")
run_nnet(df)
print(" ")
print("Cox Proportional Hazards Model Results")
run_coxph(df)
