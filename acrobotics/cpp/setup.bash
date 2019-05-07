swig -c++ -python graph.i
g++ -std=c++11 -O2 -fPIC -c src/graph.cpp graph_wrap.cxx -I /home/jeroen/anaconda3/include/python3.7m -I include
g++ -std=c++11 -O2 -shared graph.o graph_wrap.o -o _graph.so
