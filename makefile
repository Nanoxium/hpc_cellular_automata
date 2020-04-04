CPPFLAGS=-g
LDFLAGS=-g
OPENCL_LIBS=-lOpenCL -I/usr/local/cuda/include/ -L/usr/local/cuda/targets/x86_64-linux/lib/ -L/usr/local/cuda/lib64/
OPENCV_LIBS=-L/opt/opencv3/lib -lopencv_core -lopencv_highgui -I/opt/opencv3/include/
CCP=g++ $(CPPFLAGS)

parity: parity.cpp
	$(CCP) $(OPENCL_LIBS) $(OPENCV_LIBS) $< -o $@

cyclic: cyclic.cpp
	$(CCP) $(OPENCL_LIBS) $(OPENCV_LIBS) $< -o $@

clean:
	@rm -f *.o
	@rm -f cyclic
	@rm -f parity