from azureml.core import Workspace

ws = Workspace.from_config()
print("Available compute targets:")
for name in ws.compute_targets:
    ct = ws.compute_targets[name]
    print(f"  - {name}: {type(ct).__name__}")

print("\nWorkspace details:")
print(f"  Name: {ws.name}")
print(f"  Region: {ws.location}")
