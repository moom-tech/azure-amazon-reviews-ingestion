from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import AzureCliAuthentication

ws = Workspace.from_config()
datastore = Datastore.get_default(ws)

print(f"Datastore name: {datastore.name}")
print(f"Datastore type: {datastore.datastore_type}")

# List what's available in the default datastore
try:
    from azure.storage.blob import BlobServiceClient
    
    # Get connection string from the datastore
    print("\nAttempting to browse blob storage...")
    print("Default datastore container paths should contain 'cmapss-data' container")
    print("\nLet's check what datasets are already registered:")
    datasets = Dataset.get_all(ws)
    for name, dataset in datasets.items():
        print(f"\n{name}:")
        print(f"  ID: {dataset.id}")
        print(f"  Version: {dataset._version}")
        try:
            print(f"  Path reference: {dataset.data_path_references}")
        except:
            print(f"  (Could not retrieve path)")
            
except Exception as e:
    print(f"Error: {e}")
