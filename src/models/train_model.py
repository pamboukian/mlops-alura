import pickle
from tabnanny import verbose
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("train.csv")
colunas = ["LotArea", "YearBuilt", "GarageCars", "SalePrice"]
df = df[colunas]

print(df.head())

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("modelo.sav", "wb"))