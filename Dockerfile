FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["python", "src/layer3_native_nlp/entity_extractor.py"]
