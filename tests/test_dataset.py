import os, subprocess

def test_make_dataset_creates_file():
    subprocess.run(["python", "src/make_dataset.py"], check=True)
    assert os.path.exists("data/iris.csv")
