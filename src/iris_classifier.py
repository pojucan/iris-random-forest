import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar as configurações do JSON
with open('src/Hyperparameter-config.json', 'r') as f:
    config = json.load(f)

# Carregar o dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividir o dataset em treino, validação e teste
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=config["dataset"]["test_size"] + config["dataset"]["validation_size"], random_state=config["dataset"]["random_state"]
)

# Dividir o conjunto temp em validação e teste
val_size = config["dataset"]["validation_size"] / (config["dataset"]["test_size"] + config["dataset"]["validation_size"])
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=config["dataset"]["random_state"])

# Configurar o modelo com os parâmetros do JSON
model = RandomForestClassifier(
    n_estimators=config["model"]["n_estimators"],
    max_depth=config["model"]["max_depth"],
    random_state=config["model"]["random_state"]
)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Avaliar o modelo
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Acurácia do Treinamento: {train_accuracy:.2f}")
print(f"Acurácia da Validação: {val_accuracy:.2f}")
print(f"Acurácia do Teste: {test_accuracy:.2f}")

# Relatório de métricas por classe
train_report = classification_report(y_train, y_train_pred)
val_report = classification_report(y_val, y_val_pred)
test_report = classification_report(y_test, y_test_pred)

print("\nRelatório de Métricas - Treinamento:")
print(train_report)

print("\nRelatório de Métricas - Validação:")
print(val_report)

print("\nRelatório de Métricas - Teste:")
print(test_report)

# Matriz de Confusão para o conjunto de teste
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Exibindo a matriz de confusão com seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Previsões')
plt.ylabel('Valores Reais')
plt.title('Matriz de Confusão - Teste')
plt.show()
