import subprocess
from pathlib import Path

for file in Path('./data/twitter/2019').glob('*.txt'):
    subprocess.run([
        'python', 'train_word2vec.py',
        str(file), '-e', '10', '--min-word-frequency', '10'
    ], stdout=subprocess.PIPE, text=True)