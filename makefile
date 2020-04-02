CPPFLAGS=-g
LDFLAGS=-g
OPENCL_LIBS=-lOpenCL -I/usr/local/cuda/include/ -L/usr/local/cuda/targets/x86_64-linux/lib/ -L/usr/local/cuda/lib64/
OPENCV_LIBS=-L/opt/opencv3/lib -lopencv_core -lopencv_highgui -I/opt/opencv3/include/
CCP=g++ $(CPPFLAGS)

cellular_automata: ca.o
	$(CCP) $(OPENCL_LIBS) $(OPENCV_LIBS) $< -o $@

ca.o: ca.cpp
	$(CCP) $(OPENCL_LIBS) $(OPENCV_LIBS) -c $< -o $@

clean:
	@rm -f *.o
	@rm -f ca