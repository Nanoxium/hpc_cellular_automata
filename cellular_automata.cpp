
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
#include <chrono>

#include "Display.h"

using namespace std::chrono;

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

void init_square(cl_uint *domain, cl_uint2 domain_size, int size) {
    // Clearing all the domain
    for(int i = 0; i < domain_size.x; i++)
        for(int j = 0; j < domain_size.y; j++)
            domain[j * domain_size.x + i] = 0;

    int x = (domain_size.x - size)/2;
    int y = (domain_size.y - size)/2;
    
    // Drawing horizontal lines
    for(int i = 0; i <= size; i++){
        domain[y * domain_size.x + x + i] = 1;
        domain[(y + size) * domain_size.x + x + i] = 1;
    }

    // Drawing vertical lines
    for(int i = 0; i <= size; i++){
        domain[(y + i) * domain_size.x + x] = 1;
        domain[(y + i) * domain_size.x + x + size] = 1;
    }
}

int main(int argc, char** argv) {

    if (argc < 4 || atoi(argv[2]) < 2){
        std::cout << "usage : "<< argv[0] << " <kernel_file> <n> <domain_type: 0|1|2> [x size] [y size] [nb_iter] [visual] [seed]" << std::endl;
        exit(0);
    }

    cl_uint2 domain_size;
    domain_size.x = (argc >= 5) ? atoi(argv[4]) : 1000;
    domain_size.y = (argc >= 6) ? atoi(argv[5]) : 1000;
    bool visual = (argc >= 8) ? std::string(argv[7]).compare("visual")==0 : false;

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
    std::ifstream sourceFile(argv[1]);
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

    // make program out of the source code for the selected context
    cl::Program program = cl::Program(context, source);
    try {
        program.build(devices);
        cl::Kernel kernel(program, "cellular_automaton");

        // host side memory allocation
        cl_uint* domain_state = new cl_uint[domain_size.x * domain_size.y];

        // device side memory allocation for the domain
        cl::Buffer d_domain_state_a = cl::Buffer(context, CL_MEM_READ_WRITE, domain_size.x * domain_size.y * sizeof(cl_uint));
        cl::Buffer d_domain_state_b = cl::Buffer(context, CL_MEM_READ_WRITE, domain_size.x * domain_size.y * sizeof(cl_uint));

        //preparing parameters of the kernel
        cl_uint n = atoi(argv[2]);

        // Initializing a random domain
        switch(atoi(argv[3])){
            case 2:
                init_square(domain_state, domain_size, 10);
                break;
            case 1:
                std::srand((argc >= 9) ? atoi(argv[8]) : std::time(NULL)); // use current time as default random seed generator
                for(int i = 0; i < domain_size.x * domain_size.y; i++)
                    domain_state[i] = std::rand() % n;
                break;
            default:
                for(int i = 0; i < domain_size.x * domain_size.y; i++)
                    domain_state[i] = 0;
                break;
        }

        cl::NDRange local = cl::NDRange(1,1);
        cl::NDRange global = cl::NDRange(std::ceil(float(domain_size.x)/float(local[0]))*local[0], std::ceil(float(domain_size.y)/float(local[1]))*local[1]);
        std::cout << "Kernel execution" << std::endl;

        
        kernel.setArg(2, domain_size);
        kernel.setArg(3, n);

        int nb_iter = (argc >= 7) ? atoi(argv[6]) : 0;
        Display<unsigned int> d = Display<unsigned int>(domain_state, domain_size.x, domain_size.y, n);
        queue.enqueueWriteBuffer(d_domain_state_a, CL_TRUE, 0, domain_size.x * domain_size.y * sizeof(cl_uint), domain_state);
        auto start = high_resolution_clock::now();
        for(int i = 0;i < nb_iter; i++){            
            // Setting or swapping buffers during odd iteration
            if (i % 2 == 0){
                kernel.setArg(0, d_domain_state_a);
                kernel.setArg(1, d_domain_state_b);
            } else {
                kernel.setArg(1, d_domain_state_a);
                kernel.setArg(0, d_domain_state_b);
            }

            queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
            queue.finish();
            
            // Displaying current state with opencv
            if(visual){
                // Reading device buffer for the domain. This call is a synchronous call due to CL_TRUE
                queue.enqueueReadBuffer((nb_iter % 2 == 0) ? d_domain_state_a : d_domain_state_b, CL_TRUE, 0, domain_size.x * domain_size.y * sizeof(cl_uint), domain_state);
                queue.finish();
                d.show();
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop-start);
        std::cout << "Total time: " << duration.count() / 1000.0 << " [s]" << std::endl;
        if(visual) {
            d.show();
            d.waitForKey();
        }
    } catch (cl::Error error) {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
		std::string buildLog;
		program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &buildLog);
		std::cerr<<buildLog<<std::endl;
    }
    return 0;
}