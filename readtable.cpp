#include <fstream>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <vector>

int main()
{
    int16_t f;
    std::vector<int> data_real;
    std::vector<int>  data_imag;
    int n = 0;
    std::ifstream fin("usrp_samples.dat", std::ios::binary);
    while (fin.read(reinterpret_cast<char *>(&f), sizeof(int16_t))){
        //std::cout << f << '\n';
        n += 1;
        if (n%2==0)
        {
            data_real.push_back(int(f));
            std::cout << f << '\n';
        }
        else
        {
            data_imag.push_back(int(f));
        }
    }
std::cout<<data_real.size()<<std::endl;

// Create a sub vector based on read vetors
std::vector<int>::const_iterator first = data_real.begin();
std::vector<int>::const_iterator last = data_real.begin() + 41;
std::vector<int> subVec(first, last);
std::cout<<subVec.size()<<std::endl;
}