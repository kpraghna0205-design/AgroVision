import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('crop.csv')
# We train on the 4 numerical features
X = df[['Temperature', 'Rainfall', 'Fertilizer', 'Humidity']]
y = df['Yield']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, 'crop_model.pkl')
print("--- SUCCESS: AI trained with 4 features for AgroVision ---")