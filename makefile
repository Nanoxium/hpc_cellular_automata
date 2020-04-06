CPPFLAGS=-O3
LDFLAGS=-O3
OPENCL_LIBS=-lOpenCL -I/usr/local/cuda/include/ -L/usr/local/cuda/targets/x86_64-linux/lib/ -L/usr/local/cuda/lib64/
OPENCV_LIBS= -lopencv_core -lopencv_highgui -L/opt/opencv3/lib -I/opt/opencv3/include/
CCP=g++ $(CPPFLAGS)

cellular_automata: cellular_automata.cpp
	$(CCP) $(OPENCL_LIBS) $(OPENCV_LIBS) $< -o $@

clean:
	@rm -f *.o
	@rm -f cellular_automata