# AIoT-Risco-Contaminacao-Hospitalar

Sistema AIoT para classificação inteligente de risco de contaminação hospitalar utilizando sensores IoT e Machine Learning.

---
# Aplicação Online

Acesse a aplicação publicada no Streamlit:

👉 https://aiot-risco-contaminacao-hospitalar-fxdv7bwtjpnzpdbpsrynpg.streamlit.app/

---

# Objetivo do Projeto

"Este projeto foi desenvolvido como trabalho prático da disciplina de Inteligência Artificial da Pós-Graduação em Internet das Coisas (IoT) do IFSP – Câmpus Catanduva. As aulas foram ministradas pelo Prof. Dr. Marcio Andrey Teixeira."

O sistema utiliza conceitos de:

* Internet das Coisas (IoT)
* Machine Learning
* Sensores inteligentes
* Qualidade do ar
* Classificação de risco
* Streamlit
* Computação em nuvem

O objetivo é realizar a classificação inteligente de risco de contaminação no ambiente hospitalar utilizando dados de sensores e algoritmos de aprendizado de máquina.

---

# Cenário IoT

O projeto simula um ambiente hospitalar inteligente capaz de monitorar variáveis ambientais relacionadas ao risco de contaminação.

As informações podem ser obtidas por:

* Sensores reais conectados ao ESP32
* Dados simulados
* Datasets públicos de qualidade do ar

O sistema classifica o ambiente em:

* Baixo risco
* Médio risco
* Alto risco

---

# Tecnologias Utilizadas

## Hardware

* ESP32
* Sensores ambientais (BME680, PMS5003, etc.)

## Software

* Python
* Pandas
* Scikit-learn
* Streamlit
* Matplotlib
* Joblib

## Machine Learning

* Decision Tree
* Random Forest

---

# Dataset Utilizado

O projeto utiliza um dataset híbrido composto por:

* Dataset público de qualidade do ar (UCI Air Quality Dataset)
* Dados simulados gerados em Python
* Estrutura preparada para integração com sensores reais

## Variáveis utilizadas

| Variável     | Descrição                        |
| ------------ | -------------------------------- |
| temperatura  | Temperatura do ambiente          |
| umidade      | Umidade relativa do ar           |
| voc          | Compostos orgânicos voláteis     |
| co           | Monóxido de carbono              |
| nox          | Óxidos de nitrogênio             |
| no2          | Dióxido de nitrogênio            |
| particulas   | Partículas suspensas             |
| tempo_uso    | Tempo de utilização da roupa/EPI |
| area_critica | Indica se o ambiente é crítico   |
| risco        | Classificação do risco           |

---

# Modelos de Machine Learning

Foram treinados dois algoritmos:

## Decision Tree

Acurácia aproximada:

```text
95.99%
```

## Random Forest

Acurácia aproximada:

```text
97.17%
```

O modelo Random Forest apresentou melhor desempenho geral.

---

# Interface Streamlit

A aplicação Streamlit permite:

* Inserção manual dos dados dos sensores
* Classificação automática do risco
* Visualização dos resultados
* Simulação de cenários hospitalares

---

# Estrutura do Projeto

```text
AIoT-Risco-Contaminacao-Hospitalar/
│
├── app.py
├── requirements.txt
├── modelo_random_forest.pkl
├── encoder_risco.pkl
├── dataset_hibrido_risco_contaminacao.xlsx
├── README.md
└── imagens/
```

---

# Como Executar

## 1. Instalar dependências

```bash
pip install -r requirements.txt
```

## 2. Executar o Streamlit

```bash
streamlit run app.py
```

---

# Deploy

A aplicação pode ser publicada utilizando:

* Streamlit Cloud
* Render
* Hugging Face Spaces

---

# Resultados

O projeto demonstrou que técnicas de Machine Learning aplicadas à IoT podem auxiliar na classificação inteligente de risco ambiental hospitalar.

Os algoritmos apresentaram elevada acurácia, mostrando potencial para aplicações futuras em ambientes inteligentes e monitoramento hospitalar.

---

# Autor
Matheus Edilson dos Santos
Pós-Graduação em Internet das Coisas (IoT)
IFSP – Câmpus Catanduva

---

# Licença

Projeto desenvolvido para fins acadêmicos.
