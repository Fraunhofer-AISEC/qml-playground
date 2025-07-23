import logging
from logic import qml_app

logging.basicConfig(encoding='utf-8', level=logging.INFO)
qml_app.run(host="0.0.0.0")