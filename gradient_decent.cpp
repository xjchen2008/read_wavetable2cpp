#include <fstream>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <stdio.h>
//#include "matplotlibcpp.h"
#include <boost/python.hpp>
#include <Python.h>
#include <boost/python/tuple.hpp>
#include <boost/python/numpy.hpp>
namespace python = boost::python;
namespace np = boost::python::numpy;

std::vector<int> readtable(std::string filename){
    int16_t f;
    std::vector<int> data_real;
    std::vector<int>  data_imag;
    std::vector<int> data_iq;
    int n = 0;
    std::ifstream fin(filename, std::ios::binary);
    while (fin.read(reinterpret_cast<char *>(&f), sizeof(int16_t))){
        //std::cout << f << '\n';
        n += 1;
        if (n%2==0)
        {
            data_real.push_back(int(f));
            data_iq.push_back(int(f));
            //std::cout << f << '\n'; this prints the points
        }
        else
        {
            data_imag.push_back(int(f));
            data_iq.push_back(int(f));
        }
    }
    std::cout<<data_real.size()<<std::endl;

    // Create a sub vector based on read vetors
    std::vector<int>::const_iterator first = data_iq.begin();
    std::vector<int>::const_iterator last = data_iq.begin() + (4096+1)*2;//add 1 point because when PC send waveform pionts to FPGA and FPGA loads coefficient, the tlast signal is high when the last point transfered . So the last point is wasted.
    std::vector<int> subVec(first, last);
    std::cout<<subVec.size()<<std::endl;
    return subVec;
 }   
 

 
 int main(int argc, char *argv[])
{

    std::vector<int> mytaps1=readtable("/home/james/projects/leakage_cancellation/usrp_samples.dat");
    Py_Initialize();
    int D = 1000; // dimension of parameter column vector.
    float theta_real [D] ; 
    float theta_imag [D] ; 
        for(int i=0; i<D; i++){ 
            theta_real[i] = i;
            theta_imag[i] = i; 
        }
        
        python::tuple theta0 = python::make_tuple(theta_real[0], theta_imag[0]);
        python::tuple theta = python::make_tuple(theta_real[0], theta_imag[0]);
        for (int i = 0; i< D; i++){
            theta0 = python::make_tuple(theta_real[i], theta_imag[i]);
            theta = python::make_tuple(theta,theta0);
        }

    
    //python::tuple theta = python::make_tuple(theta0, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9);
    //np::ndarray example_tuple = np::array (1) ;
    std::cout << "Before gen" <<  std::endl;
    //////////////////////////
    //gen_cancellation(theta); // run the python file for generating cancellation tx signal
    //////////////////////////
    
    for (int i =0; i<= 100; i++){
        try
        {
        python::object my_python_class_module = python::import("gradient_descent_E312"); // call python and calculate the cancellation signal
        python::object main_func = my_python_class_module.attr("main")(theta, 4096, D);
        theta = python::extract<python::tuple>(main_func);
        }
        catch (const python::error_already_set&)
        {PyErr_Print();}
        std::cout << "After gen"<<  std::endl;
    }
    
    
}