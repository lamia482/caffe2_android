GXX = /opt/android-ndk-r10e/toolchains/arm-linux-androideabi-4.9/prebuilt/linux-x86_64/bin/arm-linux-androideabi-g++

SRC = intro_gl.cc
TARGET = intro_in_arm

VPATH = /home/e0024/workspace/repository/android/caffe2/tutorial/caffe2_cpp_tutorial/android_sample/arm64_/install/lib/
SHORT_STATIC_LIBRARY = libgl.a libcaffe2.a libnnpack.a libprotoc.a libprotobuf.a libpthreadpool.a libnnpack_reference_layers.a libcpufeatures.a
LONG_STATIC_LIBRARY = $(addprefix $(VPATH), $(SHORT_STATIC_LIBRARY))

default: all
	@echo Mission success, $(TARGET) is built in ./$(TARGET)

all: deps
	$(GXX) -O3 -Wl,--gc-sections -Wl,-z,nocopyreloc --sysroot=/opt/android-ndk-r10e/platforms/android-21/arch-arm -Wl,-rpath-link=/opt/android-ndk-r10e/platforms/android-21/arch-arm/usr/lib -Wl,-rpath-link=./ $(TARGET).o \
	-Wl,--whole-archive $(LONG_STATIC_LIBRARY) -Wl,--no-whole-archive \
	-L/opt/android-ndk-r10e/platforms/android-21/arch-arm/usr/lib -lGLESv3 -lGLESv2 -lGLESv1_CM -lEGL \
	/opt/android-ndk-r10e/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/thumb/libgnustl_static.a -lgcc -no-canonical-prefixes -march=armv7-a -mfpu=neon-fp16 -mfloat-abi=softfp -Wl,--fix-cortex-a8 -Wl,--no-undefined -Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -fpic -fPIE -pie -mthumb -L/opt/android-ndk-r10e/platforms/android-21/arch-arm/usr/lib -llog -lgomp -ldl -lc -lm -o $(TARGET)

deps:
	clear
	$(GXX) -O3 -MMD -MP -MF $(TARGET).o.d -fpic -fPIE -pie -ffunction-sections -funwind-tables -fstack-protector-strong -no-canonical-prefixes -march=armv7-a -mfpu=neon-fp16 -mfloat-abi=softfp -fno-exceptions -fno-rtti -mthumb -Os -g -DNDEBUG -fomit-frame-pointer -fno-strict-aliasing -finline-limit=64 -Iinstall/include -Iinstall/include/eigen -Iinstall/include/caffe2/mobile/contrib/opengl/core -I/opt/android-ndk-r10e/sources/cxx-stl/gnu-libstdc++/4.9/include -I/opt/android-ndk-r10e/sources/cxx-stl/gnu-libstdc++/4.9/libs/armeabi-v7a/include -I/opt/android-ndk-r10e/sources/cxx-stl/gnu-libstdc++/4.9/include/backward -I./ -DANDROID -std=c++11 -Wl,--whole-archive -Wa,--noexecstack -Wformat -Werror=format-security -fPIE -frtti -fexceptions  -I/opt/android-ndk-r10e/platforms/android-21/arch-arm/usr/include -c $(SRC) -o $(TARGET).o   
  
clean:
	rm -f $(TARGET) $(TARGET).o $(TARGET).o.d
