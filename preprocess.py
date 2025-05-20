### Loading and cleaning the data
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)

    #Normalizing the amnt column
    df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))

    ##Dropping the time column
    df = df.drop(["Time"], axis=1)
    return df
