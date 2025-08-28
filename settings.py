import http
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

ENVIRONMENT = os.getenv('ENVIRONMENT')

POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT= os.getenv('POSTGRES_PORT')
POSTGRES_USER= os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD= os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB_NAME= os.getenv('POSTGRES_DB_NAME')

MONGO_HOST = os.getenv('MONGO_HOST')
MONGO_PORT= os.getenv('MONGO_PORT')
MONGO_USER= os.getenv('MONGO_USER')
MONGO_PASSWORD= os.getenv('MONGO_PASSWORD')
MONGO_DB_NAME= os.getenv('MONGO_DB_NAME')

MQTT_BROKER = os.getenv('MQTT_BROKER')
MQTT_PORT = os.getenv('MQTT_PORT')
MQTT_TOPIC = os.getenv('MQTT_TOPIC')

# LOGGING
logging_level = logging.WARNING
if ENVIRONMENT in ('local', 'dev'):
    logging_level = logging.INFO
elif ENVIRONMENT == 'staging':
    logging_level = logging.INFO

logging.basicConfig(
    level=logging_level,
    format='%(asctime)s.%(msecs)03d - %(name)s : %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("BACKEND SERVICE FOR DEMO")
logging.getLogger("httpx").setLevel(logging.WARNING)
