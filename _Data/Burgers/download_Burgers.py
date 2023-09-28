import os

print("This script will download all files required for the Burgers problems."
      " You must have 'wget' installed for the downloads to work.")

folder_name = "data"
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)

fnames = [
          "burgers.tar.gz"
          ]
for fid, fname in enumerate(fnames):
    print('Downloading file {fname} ({fid+1}/{len(fnames)}):')
    url = "https://zenodo.org/record/7957915/files/" + fname
    cmd = f"wget --directory-prefix {folder_name} {url}"
    os.system(cmd)