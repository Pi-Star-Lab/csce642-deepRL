import os
import sys
import shutil
import zipfile
import subprocess
import time
import pandas as pd
from tqdm import tqdm

assignment_file = 'Q_Learning.py'
grader_csv = sys.argv[2]
submissions_path = sys.argv[1]
df = pd.read_csv(grader_csv, engine="python")
original_solution = './Solvers/' + assignment_file

shutil.move(original_solution, os.path.join("/tmp/", assignment_file))
for filename in tqdm(os.listdir(submissions_path)):
    filepath = os.path.join(submissions_path, filename)
    with zipfile.ZipFile(filepath,"r") as zip_ref:
        zip_ref.extractall("/tmp/")
    #shutil.unpack_archive(filepath, './Solvers/')
    shutil.unpack_archive(filepath, '/tmp/tmp/')
    error = False  if os.path.exists('/tmp/tmp/' + assignment_file) else True
    if error:
        from pathlib import Path
        fpath = None
        for path in Path('/tmp/tmp/').rglob('*.py'):
            if "MACOSX" in str(path):
                continue
            fpath = path
        fpath = str(fpath)
        shutil.move(fpath, "./Solvers/" + assignment_file)
    else:
        shutil.move('/tmp/tmp/' + assignment_file, "./Solvers/" + assignment_file)
    cmd = subprocess.run('timeout 10m /home/sumedhpendurkar/anaconda3/envs/csce689/bin/python autograder.py aql', \
                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = cmd.stdout.decode("utf-8").split("\n")
    points = '0'
    for line in lines:
        if "Total Points" in line:
            points = line.split(" ")[-3]
            id = filename.split("_")[1]
            if id == "LATE":
                id = filename.split("_")[2]
    #df.loc[df['ID'] == float(id), 'Value Iteration (1411281)'] = points
    print(id)
    df.loc[df['ID'] == float(id), 'Q Learning with Approximation (1439195)'] = points
    print(points)
    if not error:
        os.remove(original_solution)
    #shutil.unpack_archive(filepath, '/tmp/')
df.to_csv('updated_Grades.csv')
shutil.move(os.path.join("/tmp/", assignment_file), original_solution)
