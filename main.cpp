#include <iostream>
#include <string>
#include <vector>

#include "EnhancedKnnSparseVector.h"


using namespace std;

int main() {
    string filename = "/Users/eneskilicaslan/CLionProjects/EKNNv2/train_head.txt";
    //filename = "/Users/eneskilicaslan/Desktop/data-science/WikiLSHTC/train_head.txt";
    std::cout << "Welcome to Enhanced Knn" << std::endl;

    EnhancedKnnSparseVector eKnn(3,0.4, filename);

    eKnn.fillVectors();
    eKnn.printVectors();

    //eKnn.idf("60522");
    eKnn.printLenghts();




    return 0;
}