from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset
from azureml.core.compute import ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.datastore import Datastore

# ── Connect to workspace ─────────────────────────────────────────────────────
ws = Workspace.from_config()
compute = ComputeTarget(workspace=ws, name='lab5')

# ── Build reproducible environment ──────────────────────────────────────────
env = Environment(name='lab5-env')
conda_dep = CondaDependencies()
for pkg in ['tsfresh', 'deap', 'scikit-learn', 'xgboost',
            'lightgbm', 'pandas', 'numpy', 'matplotlib', 'seaborn']:
    conda_dep.add_pip_package(pkg)
env.python.conda_dependencies = conda_dep

# ── Register custom datastore with account key ────────────────────────────────
STORAGE_ACCOUNT = 'lab5amlworkspa4258181091'
CONTAINER = 'cmapss-data'
DATASTORE_NAME = 'cmapss_storage'

try:
    datastore = Datastore(ws, name=DATASTORE_NAME)
    print(f"Found existing datastore: {DATASTORE_NAME}")
except:
    import subprocess
    print(f"Datastore not found. Getting storage account key...")
    
    # Use shell=True to search PATH for az command
    cmd = f'az storage account keys list --account-name {STORAGE_ACCOUNT} --query "[0].value" -o tsv'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        print(f"Error running az command: {result.stderr}")
        raise RuntimeError(f"Failed to get storage account key. Make sure Azure CLI is installed and you're logged in.\nCommand: {cmd}\nError: {result.stderr}")
    
    account_key = result.stdout.strip()
    print(f"Got storage account key, registering datastore...")
    
    datastore = Datastore.register_azure_blob_container(
        workspace=ws,
        datastore_name=DATASTORE_NAME,
        container_name=CONTAINER,
        account_name=STORAGE_ACCOUNT,
        account_key=account_key
    )
    print(f"Registered datastore: {DATASTORE_NAME}")

# ── Create dataset and configure download ──────────────────────────────────
dataset = Dataset.File.from_files(path=(datastore, ''))
data_download = dataset.as_download(path_on_compute='data/raw')

# ── Configure the run ────────────────────────────────────────────────────────
config = ScriptRunConfig(
    source_directory='.',
    script='main.py',
    compute_target=compute,
    environment=env,
    arguments=[
        '--data_dir',   data_download,  # Downloads to data/raw on compute
        '--output_dir', 'outputs',      # Azure ML special folder
        '--settings',   'efficient',
        '--top_k',      '50',
        '--ga_gen',     '20',
        '--ga_pop',     '50',
    ]
)

# ── Submit ───────────────────────────────────────────────────────────────────
experiment = Experiment(workspace=ws, name='lab5-feature-pipeline')
run = experiment.submit(config)

print(f"\nRun submitted!")
print(f"Run ID: {run.id}")
print(f"Track it here: {run.get_portal_url()}\n")

# Note: Run will execute on the compute cluster
# Check results in Azure ML Studio or run check_run_status.py to monitor