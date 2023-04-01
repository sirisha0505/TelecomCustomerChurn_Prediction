FROM python:3.8-slim-buster

WORKDIR C:\Users\gnssi\Desktop\4712_b116_TelecomChurn

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY train.csv train.csv
COPY appst.py appst.py

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "appst.py", "--server.port", "8501"]