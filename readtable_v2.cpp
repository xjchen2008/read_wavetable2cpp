#include <fstream>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <python2.7/Python.h>
#include <boost/python.hpp>


template<class Container>
std::ostream& write_container(const Container& c,
                     std::ostream& out,
                     char delimiter = '\n')
{
    bool write_sep = false;
    for (const auto& e: c) {
        if (write_sep)
            out << delimiter;
        else
            write_sep = true;
        out << e;
    }
    return out;
}

int main(int argc, char *argv[])
{

/*
std::vector<short> buff(2,10);//= {{10},{10}}; 
//buff= {1,2,3};
for (std::vector<short>::const_iterator i = buff.begin(); i != buff.end(); ++i)
    std::cout << *i << ' ';
*/

// Initializing 2D vector "vect" with 
// values 
//std::vector<std::vector<int> > vect{ { 1, 2, 3 }, 
//                            { 4, 5, 6 }, 
//                            { 7, 8, 9 } }; 

// Create a 2D vector
std::vector<std::vector<int> > vect(2,std::vector<int>(10));
vect= {{1,2,3,4,5,6,7,8,9,0},{0,9,8,7,6,5,4,3,2,1}};
// Displaying the 2D vector 
for (int i = 0; i < vect.size(); i++) { 
    for (int j = 0; j < vect[i].size(); j++) 
        std::cout << vect[i][j] << " "; 
    std::cout << std::endl; 
} 
// Create a pointer for 2D vector
std::vector<int*> vect_ptrs;
for (size_t i = 0; i < vect.size(); i++) vect_ptrs.push_back(&vect[i].front());
// Display the pointer
for (std::vector<int*>::const_iterator i = vect_ptrs.begin(); i != vect_ptrs.end(); ++i)
    std::cout << *i << ' ';
std::cout<<vect_ptrs[0]<<"\n"<<&vect_ptrs<<std::endl;

// Save 2D vector to file
int num_rx_samps = 10;

std::string file = "data.dat";
std::ofstream outfile;
outfile.open(file.c_str(), std::ofstream::binary);
outfile.write((const char*)vect_ptrs[0], num_rx_samps*sizeof(int));
outfile.write((const char*)vect_ptrs[1], num_rx_samps*sizeof(int));
outfile.write((const char*)&vect[0], num_rx_samps*sizeof(int));
outfile.write((const char*)&vect[1], num_rx_samps*sizeof(int));
outfile.write((const char*)&vect[0][0], num_rx_samps*sizeof(int));
outfile.write((const char*)&vect[1][0], num_rx_samps*sizeof(int));
outfile.write((const char*)&vect[0][0], 2*num_rx_samps*sizeof(int)); // two vector is not connected in memory
outfile.close();

std::ifstream infile ("data.dat");
std::string line;
if(infile.is_open())
{
    while(getline(infile,line))
    {
        std::cout<< line<< "\n"<<std::endl;
    }
    infile.close();
}

// import python within C++

Py_Initialize();
PyRun_SimpleString("from time import time,ctime\n"
"print 'Today is',ctime(time())\n");
Py_Finalize();

}