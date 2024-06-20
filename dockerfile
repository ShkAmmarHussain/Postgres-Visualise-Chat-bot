FROM python:3.10.14

RUN pip install virtualenv

ENV VIRTUAL_ENV=/venv

RUN virtualenv venv -p python3

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

# Add application files
ADD . /app

# Install dependencies
RUN pip install -r requirements2.txt
RUN pip install yolopandas==0.0.6
RUN pip install  openai==1.30.5

# Expose port
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app2.py" , "--server.port=8501", "--server.address=0.0.0.0"]
