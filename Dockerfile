FROM python:3.10-slim

# Install necessary packages including git and build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for webapp
WORKDIR /webapp

# Copy the requirements.txt file into the container
COPY requirements.txt /webapp/

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies from GitHub repositories
RUN pip install --no-cache-dir \
    git+https://github.com/m-bain/whisperX.git@78dcfaab51005aa703ee21375f81ed31bc248560 \
    git+https://github.com/adefossez/demucs.git \
    git+https://github.com/oliverguhr/deepmultilingualpunctuation.git \
    git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

ENV LISTEN_PORT=5000
EXPOSE 5000

# Copy the rest of the application files into the container
COPY . /webapp/

# Indicate where uwsgi.ini lives
ENV UWSGI_INI uwsgi.ini

# Command to run Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
