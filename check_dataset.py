from azureml.core import Workspace, Dataset

ws = Workspace.from_config()
datasets = Dataset.get_all(ws)
print("Registered datasets:", list(datasets.keys()))
