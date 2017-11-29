chmod a+x lamia_scripts/*.sh
echo 'collecting srcs_files'
cp lamia_scripts/make_src.py caffe2/mobile/contrib/opengl/make_src.py
cd 'caffe2/mobile/contrib/opengl'
python 'make_src.py'
cd '../../../../'
echo 'done for collecting srcs_files'

srcs_file='caffe2/mobile/contrib/opengl/c_file'
echo $srcs_file
echo 'add c files into library'
for src_file in `cat $srcs_file`
do
  echo 'build '$src_file
  bash ./lamia_scripts/link_cc.sh $src_file
done

srcs_file='caffe2/mobile/contrib/opengl/cc_file'
echo $srcs_file
echo 'add cc files into library'
for src_file in `cat $srcs_file`
do
  echo 'build '$src_file
  bash ./lamia_scripts/link_xx.sh $src_file
done

NDK_PATH=/opt/android-ndk-r10e/toolchains
AR=$NDK_PATH/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-ar
TARGET='libgl.a'
echo 'linking objects into static library: 'install/lib/$TARGET

srcs_file='caffe2/mobile/contrib/opengl/o_file'
src_file=`cat $srcs_file`
existed_static_library='install/lib/libcaffe2.a install/lib/libcpufeatures.a install/lib/libnnpack.a install/lib/libnnpack_reference_layers.a install/lib/libprotobuf.a install/lib/libprotobuf-lite.a install/lib/libprotoc.a install/lib/libpthreadpool.a'

$AR rsuv install/lib/$TARGET $src_file # $existed_static_library

rm -f $src_file
