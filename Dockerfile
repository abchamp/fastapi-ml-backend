FROM python:3.9

RUN apt-get update && apt-get install -y --no-install-recommends \
  bzip2 \
  g++ \
  git \
  graphviz \
  libgl1-mesa-glx \
  libhdf5-dev \
  openmpi-bin \
  wget \
  python3-tk && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /code 
COPY ./requirements.txt /code/requirements.txt 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]