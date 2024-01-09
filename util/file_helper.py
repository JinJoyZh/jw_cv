import yaml

def get_yaml(yaml_path):
    file_yaml = yaml_path
    rf = open(file=file_yaml, mode='r', encoding='utf-8')
    crf = rf.read()
    rf.close()
    yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)
    return yaml_data

