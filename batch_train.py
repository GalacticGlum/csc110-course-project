import subprocess
from pathlib import Path

for file in Path('./data/twitter/2020/p100/').glob('*.txt'):
    print(f'Training on {file}')
    subprocess.run([
        'python', 'train_word2vec.py',
        str(file), '-e', '1', '--min-word-frequency', '10'
    ])