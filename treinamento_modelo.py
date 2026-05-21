# =========================================================
# AIoT - RISCO DE CONTAMINAÇÃO HOSPITALAR
# TREINAMENTO DO MODELO DE MACHINE LEARNING
# =========================================================

# =========================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# =========================================================

import pandas as pd
import numpy as np
import random
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# CARREGAR DATASET REAL
# =========================================================

print("Carregando dataset real...")

df_real = pd.read_csv(
    "AirQualityUCI.csv",
    sep=';',
    decimal=',',
    encoding='latin1'
)

# =========================================================
# REMOVER COLUNAS DESNECESSÁRIAS
# =========================================================

df_real = df_real.drop(
    columns=['Unnamed: 15', 'Unnamed: 16'],
    errors='ignore'
)

# =========================================================
# RENOMEAR COLUNAS
# =========================================================

df_real = df_real.rename(columns={
    "T": "temperatura",
    "RH": "umidade",
    "CO(GT)": "co",
    "NOx(GT)": "nox",
    "NO2(GT)": "no2"
})

# =========================================================
# CRIAR NOVAS VARIÁVEIS
# =========================================================

print("Criando variáveis simuladas...")

df_real["voc"] = np.random.uniform(
    50,
    700,
    len(df_real)
)

df_real["particulas"] = np.random.uniform(
    5,
    200,
    len(df_real)
)

df_real["tempo_uso"] = np.random.uniform(
    1,
    12,
    len(df_real)
)

df_real["area_critica"] = np.random.randint(
    0,
    2,
    len(df_real)
)

df_real["co2"] = np.random.uniform(
    400,
    2500,
    len(df_real)
)

# =========================================================
# FUNÇÃO DE CLASSIFICAÇÃO DE RISCO
# =========================================================

def classificar_risco(linha):

    score = 0

    if linha["temperatura"] > 30:
        score += 1

    if linha["umidade"] > 70:
        score += 1

    if linha["voc"] > 400:
        score += 1

    if linha["particulas"] > 100:
        score += 1

    if linha["tempo_uso"] > 6:
        score += 1

    if linha["area_critica"] == 1:
        score += 1

    if score <= 2:
        return "baixo"

    elif score <= 4:
        return "medio"

    else:
        return "alto"

# =========================================================
# APLICAR CLASSIFICAÇÃO
# =========================================================

print("Classificando riscos...")

df_real["risco"] = df_real.apply(
    classificar_risco,
    axis=1
)

# =========================================================
# SELECIONAR COLUNAS IMPORTANTES
# =========================================================

df_real = df_real[
    [
        "temperatura",
        "umidade",
        "voc",
        "co",
        "nox",
        "no2",
        "co2",
        "particulas",
        "tempo_uso",
        "area_critica",
        "risco"
    ]
]

# =========================================================
# GERAR DADOS SIMULADOS
# =========================================================

print("Gerando dados simulados...")

np.random.seed(42)

dados = []

quantidade = 1500

for i in range(quantidade):

    temperatura = np.random.uniform(20, 36)
    umidade = np.random.uniform(35, 90)
    voc = np.random.uniform(50, 900)

    co = np.random.uniform(0, 15)
    nox = np.random.uniform(10, 400)
    no2 = np.random.uniform(5, 200)

    co2 = np.random.uniform(400, 2500)

    particulas = np.random.uniform(5, 180)

    tempo_uso = np.random.uniform(0, 12)

    area_critica = np.random.choice([0, 1])

    pontos = 0

    if temperatura > 30:
        pontos += 1

    if umidade > 70:
        pontos += 1

    if voc > 500:
        pontos += 2

    if co2 > 1200:
        pontos += 1

    if particulas > 80:
        pontos += 2

    if tempo_uso > 6:
        pontos += 1

    if area_critica == 1:
        pontos += 1

    if pontos <= 2:
        risco = "baixo"

    elif pontos <= 5:
        risco = "medio"

    else:
        risco = "alto"

    dados.append([
        temperatura,
        umidade,
        voc,
        co,
        nox,
        no2,
        co2,
        particulas,
        tempo_uso,
        area_critica,
        risco
    ])

# =========================================================
# DATAFRAME SINTÉTICO
# =========================================================

df_simulado = pd.DataFrame(
    dados,
    columns=[
        "temperatura",
        "umidade",
        "voc",
        "co",
        "nox",
        "no2",
        "co2",
        "particulas",
        "tempo_uso",
        "area_critica",
        "risco"
    ]
)

# =========================================================
# CRIAR DATASET HÍBRIDO
# =========================================================

print("Criando dataset híbrido...")

df_hibrido = pd.concat(
    [df_real, df_simulado],
    ignore_index=True
)

# =========================================================
# SALVAR DATASET
# =========================================================

df_hibrido.to_csv(
    "dataset_hibrido_risco_contaminacao.csv",
    index=False
)

print("Dataset salvo com sucesso!")

# =========================================================
# MACHINE LEARNING
# =========================================================

print("Iniciando treinamento...")

# Variável alvo
y = df_hibrido["risco"]

# Entradas
X = df_hibrido.drop(columns=["risco"])

# =========================================================
# LABEL ENCODER
# =========================================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# =========================================================
# DIVISÃO TREINO E TESTE
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# DECISION TREE
# =========================================================

modelo_tree = DecisionTreeClassifier()

modelo_tree.fit(
    X_train,
    y_train
)

pred_tree = modelo_tree.predict(X_test)

# =========================================================
# RANDOM FOREST
# =========================================================

modelo_rf = RandomForestClassifier()

modelo_rf.fit(
    X_train,
    y_train
)

pred_rf = modelo_rf.predict(X_test)

# =========================================================
# RESULTADOS
# =========================================================

print("\n===== Decision Tree =====")

print(
    "Acurácia:",
    accuracy_score(y_test, pred_tree)
)

print(
    classification_report(
        y_test,
        pred_tree
    )
)

print("\n===== Random Forest =====")

print(
    "Acurácia:",
    accuracy_score(y_test, pred_rf)
)

print(
    classification_report(
        y_test,
        pred_rf
    )
)

# =========================================================
# GRÁFICO
# =========================================================

modelos = [
    'Decision Tree',
    'Random Forest'
]

acuracias = [
    accuracy_score(y_test, pred_tree),
    accuracy_score(y_test, pred_rf)
]

plt.figure(figsize=(8, 5))

plt.bar(
    modelos,
    acuracias
)

plt.ylabel('Acurácia')

plt.title(
    'Comparação dos Modelos'
)

plt.ylim(0.90, 1.0)

for i, acc in enumerate(acuracias):

    plt.text(
        i,
        acc + 0.002,
        f'{acc:.4f}',
        ha='center'
    )

plt.show()

# =========================================================
# SALVAR MODELOS
# =========================================================

joblib.dump(
    modelo_rf,
    "modelo_random_forest.pkl"
)

joblib.dump(
    encoder,
    "encoder_risco.pkl"
)

print("\nModelos salvos com sucesso!")