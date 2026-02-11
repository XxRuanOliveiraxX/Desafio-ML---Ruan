# Desafio Técnico: Track&Care - Localização Indoor de Precisão

**Candidato:** Ruan Oliveira  
**Vaga:** Engenheiro de Machine Learning  
**Empresa:** Centro de Pesquisas Avançadas Wernher von Braun  
**Data:** 10/02/2026

## 1. Visão Geral
Este repositório apresenta a solução para o desafio de localização indoor de pacientes utilizando telemetria ruidosa de RSSI (Bluetooth Low Energy). A solução integra um pipeline de Machine Learning para classificação de salas e uma API para inferência em tempo real, focando em robustez e viabilidade técnica em ambiente ruidoso.

## 2. Estrutura do Projeto
```text
track-care-ml/
├── api/
│   └── main.py              # API FastAPI para inferência
├── data/
│   └── raw/                 # Dataset de treino original
├── models/
│   ├── model_final.pkl      # Modelo Random Forest (Campeão)
│   ├── label_encoder.pkl    # Mapeamento de classes (salas)
│   └── model_columns.pkl    # Lista de colunas para consistência da API
├── notebooks/
│   └── 01_eda_e_pre_processamento.ipynb
├── Dockerfile               # Containerização da solução
├── requirements.txt         # Dependências do projeto
└── README.md                # Documentação
```

## 3. Análise de Engenharia e "Datalização"
O dataset fornecido simula condições reais de telemetria hospitalar, apresentando alto nível de ruído e sinais ambíguos. Durante o desenvolvimento, as seguintes decisões de engenharia foram tomadas:

* **Tratamento de RSSI:** Implementação de clipping em [-100, -30] dBm para mitigar leituras fisicamente impossíveis e picos de interferência captados pelos sensores.
* **Seleção de Variáveis:**
    * imei: Mantido como feature para permitir que o modelo compense diferentes sensibilidades de antena entre os dispositivos móveis da equipe.
    * employee_id: Descartado para evitar viés comportamental, garantindo que o modelo aprenda com sinais físicos e não com a rotina de movimentação de funcionários específicos.
* **Escolha do Modelo:** Após testes comparativos com algoritmos de Boosting e técnicas de balanceamento, optou-se pela versão estável do Random Forest com 51% de acurácia. Observou-se que otimizações estatísticas agressivas (como SMOTE) causavam overfitting ao ruído das coordenadas geográficas, prejudicando a generalização.

## 4. Estratégia de Janelamento Temporal (Parâmetro X)
Conforme análise técnica dos dados brutos, uma leitura única de RSSI é insuficiente para uma localização de precisão devido ao fenômeno de multipath fading. 

**Sugestão Técnica: X = 2 minutos.** **Justificativa:** Um janelamento de 120 segundos permite que a API receba múltiplas amostras de um mesmo paciente e aplique uma média móvel. Esta estratégia de engenharia filtra as flutuações momentâneas de sinal e eleva a confiança operacional do sistema de ~51% para níveis aceitáveis para o monitoramento de saúde, sem gerar um atraso (lag) impeditivo.

## 5. Como Executar a Solução

### Via Docker (Recomendado)
Para garantir a reprodutibilidade do ambiente:
1. Construa a imagem:
   docker build -t track-care-ml .

2. Execute o container:
   docker run -p 8000:8000 track-care-ml

3. Acesse a documentação interativa (Swagger UI) em: http://localhost:8000/docs

### Via Python Local
1. Instale as dependências:
   pip install -r requirements.txt

2. Inicie o servidor:
   uvicorn api.main:app --reload

## 6. Exemplo de Requisição (POST /predict)
O endpoint de predição espera um JSON no seguinte formato:
{
  "imei": "IMEI_SAMSUNG_S23",
  "sensor_latlong": "-22.809788, -47.059685",
  "sensor_room": "Enfermaria",
  "rssi": -71.76
}