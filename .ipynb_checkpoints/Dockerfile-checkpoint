# 1. Baixa uma imagem do Linux já com Python 3.11 instalado
FROM python:3.11-slim

# 2. Define que, dentro do "computador virtual", trabalharemos na pasta /app
WORKDIR /app

# 3. Copia o seu arquivo de dependências para dentro do container
COPY requirements.txt .

# 4. Instala todas as bibliotecas (pandas, sklearn, fastapi, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia as suas pastas de código e modelos para dentro do container
COPY api/ ./api/
COPY models/ ./models/

# 6. Avisa ao Docker que a API vai usar a porta 8000
EXPOSE 8000

# 7. O comando que inicia a API automaticamente quando o container ligar
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]