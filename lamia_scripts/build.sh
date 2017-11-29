set -e
chmod a+x ./lamia_scripts/compile.sh
bash ./lamia_scripts/compile.sh
echo 'waiting for library generate...'
chmod a+x ./lamia_scripts/add_gl_library.sh
bash ./lamia_scripts/add_gl_library.sh
echo 'All tasks done, installed in ./install'