%module graph

%{
    #define SWIG_FILE_WITH_INIT
    #include "include/graph.h"
%}

%include "numpy.i"

%init %{
  import_array();
%}

// apply numpy typemaps for input stuff
//%apply (float* IN_ARRAY1, int DIM1) {(float* vec, int n)}
// IMPORTANT numpy input array must have dtype = 'float32'
%apply (float* IN_ARRAY2, int DIM1 , int DIM2){(float* mat, int nrows, int ncols)}

%apply ( int* ARGOUT_ARRAY1, int DIM1 ){(int* vec, int n)}

%ignore Node;
%ignore sort_function(Node*, Node*);
%include "include/graph.h"