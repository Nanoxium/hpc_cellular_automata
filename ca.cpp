
#define __CL_ENABLE_EXCEPTIONS

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

#include "Display.h"

void showPlatforms(){
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    for(int i = 0; i < platforms.size(); i++) {
        std::cout<<"Platform: "<<platforms[i].getInfo<CL_PLATFORM_NAME>()<<std::endl;
        std::vector<cl::Device> devices;
        platforms[i].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
        for(int j = 0; j < platforms.size(); j++) {
            if(devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CPU) std::cout << "Device: " << " CPU " << " : "<< devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << " compute units "<<"( "<<devices[j].getInfo<CL_DEVICE_NAME>()<<" )"<<std::endl;
            if(devices[j].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) std::cout << "Device: " << " GPU " << " : "<< devices[j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << " compute units "<<"( "<<devices[j].getInfo<CL_DEVICE_NAME>()<<" )"<<std::endl;
        }
    }
}

cl::Context getContext(cl_device_type requestedDeviceType, std::vector<cl::Platform>& platforms) {
    for(unsigned int i = 0; i < platforms.size(); i++) {
        try{
            cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[i])(), 0};
            return cl::Context(requestedDeviceType, cps);
        } catch (...) {}
    }
    throw CL_DEVICE_NOT_AVAILABLE;
}

int main(int argc, char** argv) {

    if (argc < 2)
        std::cout << "usage : cellular_automata <n>" << std::endl;

    cl_uint2 domain;
    domain.x = 1000;
    domain.y = 1000;

    showPlatforms();

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // gets the context 
    cl::Context context = getContext(CL_DEVICE_TYPE_GPU, platforms);

    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    std::string deviceName;
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

    std::cout << "Command queue created succesfuly, kernels will be executed on :" << deviceName << std::endl;

    
    // Reading source code for the
    std::ifstream sourceFile("ca.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    // make program out of the source code for the selected context
    cl::Program program = cl::Program(context, source);
    try {
        program.build(devices);

        cl::Kernel kernel(program, "cyclic_cellular_automata");

        // host side memory allocation
        cl_uint* domain_state = new cl_uint[domain.x * domain.y];

        // device side memory allocation for the domain
        cl::Buffer d_domain_state = cl::Buffer(context, CL_MEM_READ_WRITE, domain.x * domain.y * sizeof(cl_uint));

        //preparing parameters of the kernel
        cl_uint n = atoi(argv[1]);

        // Initializing a random domain
        for(int i = 0; i < domain.x * domain.y; i++)
            domain_state[i] = rand() % n;

        cl::NDRange local = cl::NDRange(1,1);
        cl::NDRange global = cl::NDRange(std::ceil(float(domain.x)/float(local[0]))*local[0], std::ceil(float(domain.y)/float(local[1]))*local[1]);
        std::cout << "Kernel execution" << std::endl;

        kernel.setArg(0, d_domain_state);
        kernel.setArg(1, domain);
        kernel.setArg(2, n);

        Display<unsigned int> d(domain_state, domain.x, domain.y, n);
        queue.enqueueWriteBuffer(d_domain_state, CL_TRUE, 0, domain.x * domain.y * sizeof(cl_uint), domain_state);
        for(int i = 0; i < 100000; i++){
            // std::cout << "Iteration: " << i << std::endl;
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

            // Reading device buffer for the domain. This call is a synchronous call due to CL_TRUE
            if (i > 50000)
                queue.enqueueReadBuffer(d_domain_state, CL_TRUE, 0, domain.x * domain.y * sizeof(cl_uint), domain_state);
            queue.finish();

            // Displaying current state with opencv
            if(i > 50000)
                d.show();
        }
        d.waitForKey();
    } catch (cl::Error error) {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
		std::string buildLog;
		program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &buildLog);
		std::cerr<<buildLog<<std::endl;
    }
    return 0;
}