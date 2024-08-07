# base image
FROM python:3.10.12-slim

# work folder
WORKDIR /app

# copy file in container
COPY requirements.txt .

# # setup latest pip
# RUN pip install --upgrade pip

#setup
RUN pip install --no-cache-dir -r requirements.txt

# copy project in container
COPY . .

# run script to download file pre-trained
RUN python download_pre_trained.py

# CLI run fastapi
CMD ["python", "main.py"]