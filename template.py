import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    'src/__init__.py',
    'src/helper.py',
    'src/prompt.py',
    '.env',
    'setup.py',
    'experiment/exp.ipynb',
    'app.py',
    'store_index.py',
    'static/style.css',
    'templates/chat.html'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    file_dir, file_name = os.path.split(filepath)
    
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f'Making directory: ({file_dir} for the file: ({file_name}))')

    if filepath.suffix == '':  # It's a directory like 'static'
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
            logging.info(f'Creating empty directory: {filepath}')
        continue

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
        logging.info(f'Creating empty file: {file_name}')
    else:
        logging.info(f'{file_name} is already created.')
