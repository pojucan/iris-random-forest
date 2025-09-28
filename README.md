## Iris Classifier

### Descrição
Implementação de um classificador Random Forest para o dataset Iris com configuração via JSON. Esse código visa a validação para outros tipos de dados e testes manuais de hiperparâmetros.

### Instalação

1. Clone o repositório:
```bash
git clone [url-do-repositorio]
cd projeto-iris
```
### Crie um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### Instale as dependências:

```bash
pip install -r requirements.txt
```

### Uso:

#### Execute o cássificador:

```bash
python src/iris_classifier.py
```

### Estrutura do Projeto:

```src/``` - Código fonte

```src/Hyperparameter-config.json``` - Configurações do modelo

```requirements.txt``` - Dependências do projeto

### Configuração:

Edite ```src/Hyperparameter-config.json``` para modificar parâmetros do modelo.

### Resultados:

O projeto gera:

- Métricas de acurácia (treino/validação/teste)

- Relatórios de classificação

- Matriz de confusão visual

