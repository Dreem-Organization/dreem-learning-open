FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime


ENV PYTHONPATH=/app:$PYTHONPATH



WORKDIR /app
COPY . .

RUN pip install pip numpy --upgrade \
    && pip install -r /app/requirements.txt \
    && pip install /app \
    && cp dreem_learning_open/settings_template.py dreem_learning_open/settings.py
