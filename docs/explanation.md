

# Explicação do Código

## Dataset Iris
- 150 amostras de flores Iris
- 3 classes: setosa, versicolor, virginica
- 4 features: comprimento e largura da sépala e pétala

## Configuração via JSON
O projeto usa um arquivo JSON para configuração, permitindo:
- Ajuste de tamanhos dos conjuntos de dados
- Configuração de hiperparâmetros do modelo
- Reprodutibilidade via random states

## Modelo Random Forest
- Ensemble de árvores de decisão
- Reduz overfitting através de bagging
- Hiperparâmetros configuráveis:
  - `n_estimators`: Número de árvores
  - `max_depth`: Profundidade máxima das árvores

## Avaliação
O modelo é avaliado em três conjuntos:
1. **Treino**: Ajuste do modelo
2. **Validação**: Ajuste de hiperparâmetros
3. **Teste**: Avaliação final do modelo

## Métricas
- Acurácia geral
- Relatório por classe (precision, recall, f1-score)
- Matriz de confusão visual
