//
// Created by Enes Kılıçaslan on 15/07/17.
//

#ifndef EKNNV3_ENHANCEDKNNSPARSEVECTOR_H
#define EKNNV3_ENHANCEDKNNSPARSEVECTOR_H

#include <string>
#include <vector>

class EnhancedKnnSparseVector {

public:
    EnhancedKnnSparseVector(const std::string &inputFileName);

    void fillVectors();
    void printVectors() const;

//   float similarityBM25() const ;
    int n(std::string w) const; // TODO will be private
    double idf(std::string w) const;// TODO will be private
    double fPrime(std::string w, std::vector<int> s) const;

    void printLenghts() const;



private:

    int docCounter;

    std::string inputFileName;

    std::vector< std::vector < std::string > > labels; //labels for each document
    std::vector< std::string > words; //column, all of the words in the corpus
    std::vector< std::vector < int > > docs; //(0, vector<int>(0)); //row, vector for each document | each initalized to 0
    std::vector< int > lenghts; //lengths of the training documents

    const float K1 = 0.25, b = 0.75; //constants for BM25 similarity
};


namespace EnesKilicaslanCommonOperations{

    template <typename T, typename A>
    int contains(std::vector<T, A> const *v, T x);

    template <typename T>
    std::string  numberToString(T pNumber);

}







#endif //EKNNV3_ENHANCEDKNNSPARSEVECTOR_H
