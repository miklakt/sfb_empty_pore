import os
import shutil
import re

# Change working directory to the script's location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Paths
tex_file = 'Diffusion_through_nanopore_SI.tex'
figures_source_folder = './fig/'
figures_target_folder = './used_fig/'

# Ensure target folder exists
os.makedirs(figures_target_folder, exist_ok=True)

# Read the content of the .tex file
with open(tex_file, 'r') as file:
    tex_content = file.read()

# Find all included graphics using regex
used_figures = re.findall(r'\\includegraphics.*?\{(.+?)\}', tex_content)
used_figures = set([os.path.basename(f) for f in used_figures])  # Get unique basenames

# Copy used figures
for figure in used_figures:
    source_path = os.path.join(figures_source_folder, figure)
    target_path = os.path.join(figures_target_folder, figure)

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)
        print(f'Copied: {figure}')
    else:
        print(f'Warning: {figure} not found in source folder.')