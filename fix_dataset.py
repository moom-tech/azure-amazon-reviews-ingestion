from azureml.core import Workspace, Dataset

ws = Workspace.from_config()

# Delete the broken dataset version
print("Deleting broken dataset...")
try:
    dataset = Dataset.get_by_name(ws, name='cmapss-data')
    dataset.unregister_all_versions()
    print("Deleted cmapss-data")
except Exception as e:
    print(f"Dataset not found or error: {e}")

# Register correctly from the blob container
from azureml.core.datastore import Datastore

# Get the default datastore (which points to the workspace blob storage)
datastore = ws.get_default_datastore()
print(f"\nUsing datastore: {datastore.name}")
print(f"Type: {datastore.datastore_type}")

# Register dataset pointing to the cmapss-data FOLDER in the datastore
# (where we uploaded our files earlier with az storage blob upload-batch)
dataset = Dataset.File.from_files(
    path=(datastore, 'cmapss-data')  # Path within the workspace blob storage
)
dataset.register(
    workspace=ws,
    name='cmapss-data',
    description='CMAPSS train/test/RUL data files',
    create_new_version=True
)

print("\nDataset registered successfully!")
print("Now test locally with: python main.py --data_dir data/raw")
