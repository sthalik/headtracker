#!/bin/sh -e

if uname -s | fgrep -iq mingw; then
	MINGW="-DMINGW"
	VER="242"
	SUFFIX=.exe
else
	VER=""
	MINGW=""
	SUFFIX=""
fi

LIBS="-lopencv_calib3d$VER -lopencv_contrib$VER -lopencv_core$VER -lopencv_features2d$VER -lopencv_flann$VER -lopencv_highgui$VER -lopencv_imgproc$VER -lopencv_legacy$VER -lopencv_ml$VER -lopencv_nonfree$VER -lopencv_objdetect$VER -lopencv_photo$VER -lopencv_stitching$VER -lopencv_video$VER -lopencv_videostab$VER"
LDFLAGS="-L../thirdparty/bin -L/usr/local/lib -Wl,-rpath,."
CXXFLAGS="-I../../headtracker -I../thirdparty/include -I/usr/local/include -O3 -march=native -ffast-math"

cd "$(dirname -- "$0")"
mkdir build
cd build

for i in ../../headtracker/*.cpp ../../headtracker-demo/headtracker-demo.cpp; do
	echo CXX $(basename "$i" .cpp)
	g++ $MINGW -c -o "$(basename "$i" .cpp).o" "$i" $CXXFLAGS || exit 1
done

echo LD headtracker-demo
g++ -o headtracker-demo$SUFFIX *.o $LIBS $LDFLAGS
cp ../../thirdparty/bin/head.raw .
cp ../../thirdparty/bin/*.xml .
if test -n "$MINGW"; then
	cp ../thirdparty/bin/* .
	cp /mingw/bin/libgcc_s_dw2-1.dll .
	cp /mingw/bin/libstdc++-6.dll .
fi
