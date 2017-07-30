#include <iostream>
#include <string>
#include <vector>

#include "EnhancedKnnSparseVector.h"


using namespace std;

int main() {
    const string filename = "/Users/eneskilicaslan/CLionProjects/EKNNv2/train_head.txt";
    std::cout << "Welcome to Enhanced Knn" << std::endl;

    EnhancedKnnSparseVector eKnn(2,1.5, filename);

    eKnn.fillVectors();
    eKnn.printVectors();

    //eKnn.idf("60522");
    cout << " Lengths: " << endl;
    eKnn.printLenghts();




    return 0;
}