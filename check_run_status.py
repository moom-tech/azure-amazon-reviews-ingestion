from azureml.core import Workspace, Experiment

# Connect to workspace
ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name='lab5-feature-pipeline')

# List recent runs
all_runs = list(experiment.get_runs())
runs = [r for r in all_runs if r.status == 'Running']  # Filter running runs
if not runs:
    runs = all_runs[:5]  # Show last 5 runs if none are running

print(f"\nRecent runs in 'lab5-feature-pipeline' experiment:")
print("=" * 80)

for run in runs:
    print(f"\nRun ID: {run.id}")
    print(f"Status: {run.status}")
    print(f"Portal URL: {run.get_portal_url()}")
    
    # Show status
    if run.status == 'Finished':
        print(f"✓ Run completed successfully")
    elif run.status == 'Running':
        print(f"→ Run is currently executing on compute cluster 'lab5'")
    elif run.status == 'Failed':
        print(f"✗ Run failed. Check Azure ML Studio for error details.")
