# your_script.py
import sys
import os
import json
from PyQt6.QtWidgets import QApplication
from gui_main import Main

if __name__ == '__main__':
    if not os.path.exists("settings.json"):
        with open('settings.json', 'w') as fp:
            json.dump({"default_directory": os.getcwd()}, fp)
    
    app = QApplication(sys.argv)
    m = Main()
    m.show()
    sys.exit(app.exec())