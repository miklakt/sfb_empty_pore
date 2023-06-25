#%%
import pathlib

dir = pathlib.Path("temp8")

#%%
for i, file in enumerate(dir.glob("*.in")):
    filename = str(file.stem)
    print(filename)
    with open(dir/f"job_{i}.sh", "w") as sbatch_file:
        sbatch_file.write(
f"""#!/bin/bash
# Generated file
#SBATCH --partition=72hours
#SBATCH --qos=72hours
#SBATCH --job-name={filename}
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --exclusive
#SBATCH --err={filename}.err
#SBATCH --output={filename}.log
#----------------------------------------------------------#
"""+\
"""
echo "The job "${SLURM_JOB_ID}" is running on "${SLURM_JOB_NODELIST}
"""+\
f"""
#----------------------------------------------------------#
srun --ntasks=1 --hint=nomultithread ./sfbox {filename}.in     
"""
        )
# %%
import os
os.popen(f'cp ~/.local/bin/sfbox {dir.absolute()}/sfbox') 
# %%
with open(dir/f"run_jobs", "w") as sbatch_run:
    sbatch_run.write(
"""#!/bin/bash
# Generated file
#!/bin/bash
for f in *.sh
do
        sbatch "$f"
done
"""
    )