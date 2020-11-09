// Use the following command to compile
//gcc py.cpp -o py -lpython2.7 -lm -L/usr/lib/python2.7/config/ -I/usr/include/python2.7/

#include <stdio.h>
#include <Python.h>
#include <pythonrun.h>
int main(int argc, char *argv[]) {
Py_Initialize();
PyRun_SimpleString("from time import time,ctime\n"
"print 'Today is',ctime(time())\n");
Py_Finalize();
return 0;
}

