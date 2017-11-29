import os
import sys

required_src = ['operators', 'android', 'core', 'test']

def list_dir(path):
  return os.listdir(path)
  
def backend(file):
  return os.path.splitext(file)[1] 
  
def open_file(filename):
  if os.path.exists(filename):
    os.remove(filename)
  f = open(filename, 'w')
  return f
  
def add_to_file(folder, file, fp, filename, by_line = True):
  full_file = os.path.join(folder, file)
  dispath = ' '
  if by_line is True:
    dispath = '\n'
  fp.writelines(full_file + dispath)
  print('file: {} add to {}'.format(full_file, filename))
  
def save_file_name(prefix, folders):
  c_file = 'c_file'
  cc_file = 'cc_file'
  o_file = 'o_file'
  c_fp = open_file(os.path.join(prefix, c_file))
  cc_fp = open_file(os.path.join(prefix, cc_file))
  o_fp = open_file(os.path.join(prefix, o_file))
  for folder in folders:
    if not os.path.isdir(folder):
      continue
    if folder not in required_src:
      continue
    src_path = os.path.join(prefix, folder)
    files = list_dir(src_path)
    for file in files:
      if backend(file) == '.c':
        add_to_file(folder, file, c_fp, c_file)
        add_to_file(os.path.join(prefix, folder), file + '.o', o_fp, o_file, False)
      elif backend(file) == '.cc':
        add_to_file(folder, file, cc_fp, cc_file)
        add_to_file(os.path.join(prefix, folder), file + '.o', o_fp, o_file, False)

if __name__ == '__main__':
  prefix = './'
  gl_dir = list_dir(prefix)
  save_file_name(prefix, gl_dir)
