from azureml.core import Workspace, Dataset

ws = Workspace.from_config()
print("Registered datasets:")
try:
    for name in ws.datasets:
        print(f"  - {name}")
except Exception as e:
    print(f"  (Error: {e})")
