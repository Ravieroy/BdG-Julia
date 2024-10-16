import importlib.util

library_list = [
    "numpy",
    "matplotlib",
    "seaborn",
    "pickle",
    "benedict",
    "warnings",
]

installed_packages = library_list.copy()
for name in library_list:
    val = importlib.util.find_spec(name)
    if val is None:
        installed_packages.remove(name)


missing_packages = list(set(library_list) - set(installed_packages))
if missing_packages:
    print("missing Packages: ")
    for name in missing_packages:
        print(name)
else:
    print("All dependencies are satisfied")
