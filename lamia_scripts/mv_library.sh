cp build_android/confu-deps/libcpufeatures.a ./install/lib
cp build_android/confu-deps/libnnpack_reference_layers.a ./install/lib
cp -r install ./tutorial_arm64_v7a
cp -r third_party/eigen ./tutorial_arm64_v7a/install/include
echo 'copy library and header file done'
