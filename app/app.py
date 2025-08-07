import logging
from logic import qml_app

logging.basicConfig(encoding='utf-8', level=logging.INFO)

server = qml_app.server

if __name__ == '__main__':
    qml_app.run(host="0.0.0.0")