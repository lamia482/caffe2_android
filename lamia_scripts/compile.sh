rm -rf build_*
rm -rf install
chmod a+x ./scripts/build_android.sh
chmod a+x ./scripts/build_host_protoc.sh
bash ./scripts/build_android.sh
cd build_android && make install -j32
