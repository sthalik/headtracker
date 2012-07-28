#!/bin/sh -e

MINGW=""

if uname -s | fgrep -iq mingw; then
	MINGW="-DMINGW"
fi

LIBS="-lopencv_calib3d242 -lopencv_contrib242 -lopencv_core242 -lopencv_features2d242 -lopencv_flann242 -lopencv_gpu242 -lopencv_highgui242 -lopencv_imgproc242 -lopencv_legacy242 -lopencv_ml242 -lopencv_nonfree242 -lopencv_objdetect242 -lopencv_photo242 -lopencv_stitching242 -lopencv_video242 -lopencv_videostab242 -lopencv_ffmpeg242"
LDFLAGS="-L../thirdparty/bin -Wl,-rpath,."
CXXFLAGS="-I../../headtracker -I../thirdparty/include -O3 -march=native -ffast-math"

cd "$(dirname -- "$0")"
mkdir build
cd build

for i in ../../headtracker/*.cpp ../../headtracker-demo/headtracker-demo.cpp; do
	echo CXX $(basename "$i" .cpp)
	g++ $MINGW -c -o "$(basename "$i" .cpp).o" "$i" $CXXFLAGS || exit 1
done

echo LD headtracker-demo
g++ -o headtracker-demo.exe *.o $LIBS $LDFLAGS
cp ../../thirdparty/bin/head.raw .
cp ../../thirdparty/bin/*.xml .
if test -n "$MINGW"; then
	cp ../thirdparty/bin/* .
	cp /mingw/bin/libgcc_s_dw2-1.dll .
	cp /mingw/bin/libstdc++-6.dll .
fi
