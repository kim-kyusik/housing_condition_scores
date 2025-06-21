# logger.py

def create_log(content, file_path, mode):
    with open(file_path, mode) as f:
        f.write(content)