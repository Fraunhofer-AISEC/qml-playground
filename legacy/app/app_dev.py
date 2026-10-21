import argparse
import logging

from logic import qml_app

# Configuration
argparser = argparse.ArgumentParser(description="QML Playground")
argparser.add_argument(
    "--is_debug", type=bool, default=True, help="Debug the code or deploy on the server"
)
args = argparser.parse_args()

if __name__ == '__main__':

    logging.basicConfig(encoding='utf-8', level=logging.INFO)

    if args.is_debug:
        qml_app.run(debug=True)
    else:
        qml_app.run(debug=False)
