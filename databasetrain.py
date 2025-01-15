import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
df=pd.read_csv('Test_Vision_Final_Org_2.csv')
#df.head()
x=df.drop('clas',axis=1)
y=df['clas']
# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Optional: Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)

# Train the model
mlp.fit(X_train, y_train)

# Make predictions
y_pred = mlp.predict(X_test)
# Evaluate the model
accuracy = mlp.score(X_test, y_test)
print(f"Accuracy: {accuracy}")
results = pd.DataFrame({
    'Expected': y_test,
    'Predicted': y_pred
})

# Mostrar los resultados
print(results)
#Guardar el modelo
#joblib.dump(mlp, 'mlp_mode_Paselistal23.pkl')
#joblib.dump(scaler, 'scaler_paselista2.pkl')
#Matriz de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()