import os
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO,format = '[%(asctime)s]: %(message)s:')

project_name = "personalAssistant"

list_of_files = [
    ".github/workflows/.gitkeep",
    "backend/ingest_data.py",
    "backend/model.py",
    "static/app.js",
    "static/style.css",
    "app.py",
    ".env",
    "Dockerfile"
    "requirements.txt",
    "templates/index.html",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir,exist_ok = True)
        logging.info(f"creating directory : {filedir} ")
    

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"creating empty file: {filepath}")



    else:
         logging.info(f"{filename} is already exists")       