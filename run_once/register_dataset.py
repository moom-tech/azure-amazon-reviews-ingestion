from azureml.core import Workspace, Dataset, Datastore

ws = Workspace.from_config()

# The default datastore connected to your blob storage
datastore = Datastore.get(ws, 'workspaceblobstore')

# Upload from local if not already done via CLI
datastore.upload(
    src_dir='data/raw',
    target_path='cmapss-data',
    overwrite=False
)

# Register as a versioned dataset
dataset = Dataset.File.from_files(path=(datastore, 'cmapss-data/'))
dataset = dataset.register(
    workspace=ws,
    name='cmapss-fd001',
    description='NASA C-MAPSS FD001 turbofan engine degradation dataset',
    create_new_version=True
)
print(f"Registered: {dataset.name} v{dataset.version}")