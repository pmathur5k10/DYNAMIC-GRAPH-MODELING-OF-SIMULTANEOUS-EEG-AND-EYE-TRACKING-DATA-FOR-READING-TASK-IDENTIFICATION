import zipfile
with zipfile.ZipFile('../../data/ST.zip', 'r') as zip_ref:
    zip_ref.extractall('../../data/')