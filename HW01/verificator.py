import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar los datos de entrenamiento
train_df = pd.read_csv('train.csv')

# Separar las características (X) y el target (y)
train_x = train_df.drop("Performance Index", axis=1)
train_y = train_df["Performance Index"]

# Crear el modelo de scikit-learn para regresión lineal
model = LinearRegression()

# Ajustar el modelo a los datos
model.fit(train_x, train_y)

# Obtener los coeficientes y el intercepto
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')
