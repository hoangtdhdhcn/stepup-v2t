FROM python:3.10-slim

ENV LISTEN_PORT=5000
EXPOSE 5000

# Indicate where uwsgi.ini lives
ENV UWSGI_INI uwsgi.ini

# Set the working directory in the container
WORKDIR /webapp

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD [ "python", "./app.py" ]
