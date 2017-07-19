//
// Created by Enes Kılıçaslan on 15/07/17.
//


#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>


#include "EnhancedKnnSparseVector.h"
using namespace std;

EnhancedKnnSparseVector::EnhancedKnnSparseVector(const string &inputFileName)
        : inputFileName(inputFileName) {

    this->docCounter = 0;
}

void EnhancedKnnSparseVector::fillVectors() {

    ifstream input(inputFileName);
    string line;
    getline( input, line ); // skip header

    while( getline( input, line ) ) {

        if(line == "\n")
            continue; //skip empty lines

        vector<string> labels_local;
        vector<int> docs_local(words.size(), 0);

        stringstream ss_main(line); // Turn the string into a stream.
        string tok_main;
        getline(ss_main, tok_main, ' ');
        string tok_lab;
        stringstream ss_lab(tok_main);
        //cout <<"debug 1" << endl;

        while( getline(ss_lab, tok_lab, ',')){
            labels_local.push_back(tok_lab);
        }

        labels.push_back(labels_local); //append labels
        docs.push_back(docs_local);
        lenghts.push_back(0); //fist initialize lenght of the document to zero

        //parse word and occurance
        while(getline(ss_main, tok_main, ' ')){
            //cout <<"debug 2" << endl;

            long int pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance = stoi( tok_main.substr(pos + 1, 1) );
            int word_index = EnesKilicaslanCommonOperations::contains(&words, word);

            if (word_index >= 0){
                //cout << "index Debug " << word << " " << word_index << "docs counter " << doc_counter <<  endl;
                //cout << "lend docs: " << docs.size() << endl;
                docs[this->docCounter][word_index] = occurance;
                lenghts[this->docCounter] += occurance;

            } else {
                words.push_back(word);

                //TODO make it with -1 for efficiency ! And use push back for occurances
                //!! Attention -- this makes really sparse matrix
                for( int i=0; i < docs.size(); ++i) //because this is new word, it never occurred before
                    docs[i].push_back(0);  // so make it 0 for the other documents

                docs[this->docCounter][words.size() - 1] = occurance; // Bag of words (not binary)
                lenghts[this->docCounter] += occurance;
            }
        }

        //cout <<"debug 3" << endl;
        this->docCounter += 1;
    }

}

void EnhancedKnnSparseVector::printVectors() const {

    for(vector<string>::const_iterator it = words.begin(); it != words.end(); ++it) {
        printf("%8s | ", (*it).c_str());
    }

    cout << endl;
    std::vector< std::vector<int> >::const_iterator row;
    std::vector<int>::const_iterator col;

    for (row = this->docs.begin(); row != this->docs.end(); ++row)
    {
        for (col = row->begin(); col != row->end(); ++col)
        {

            printf("%8s | ", EnesKilicaslanCommonOperations::numberToString(*col).c_str());
        }

        cout << endl;
    }

}

double EnhancedKnnSparseVector::fPrime(std::string w, std::vector<int> s) const {
    int fws;

    int indx = EnesKilicaslanCommonOperations::contains(&words, w);
    //check if the word exist in our corpus
    //we are almost sure that it will exist, but security first!
    if (indx == -1) //TODO think about these return -1's again
        return -1;

    fws = s[indx];


    return ((K1 + 1) * fws) / (K1 + (1 -b));







}


/**
*                   N - n(w) + 0.5
*  idf(w) = log ----------------------
*                     n(w) + 0.5
* */
double EnhancedKnnSparseVector::idf(std::string w) const {
    int nw = n(w);

    return log((docCounter - nw + 0.5) / (nw + 0.5));
}

int EnhancedKnnSparseVector::n(std::string w) const {
    int result = 0;

    int indx = EnesKilicaslanCommonOperations::contains(&words, w);
    //check if the word exist in our corpus
    //we are almost sure that it will exist, but security first!
    if (indx == -1)
        return -1;

    for(int i=0; i< docCounter; ++i) {
        int occ = docs[i][indx];
        if ( occ > 0)
        result += occ;
    }

    return result;
}


void EnhancedKnnSparseVector::printLenghts() const {
    for(int i=0; i < docCounter; ++i)
        cout << lenghts[i] << endl;
}


namespace EnesKilicaslanCommonOperations{

    template <typename T, typename A>
    int contains(vector<T, A> const *v, T x){
        typename vector<T, A>::const_iterator it = find(v->begin(), v->end(), x); //iterator

        if (it == v->end())
            return -1;
        else
            return it - v->begin();
    }

    template <typename T>
    string  numberToString(T pNumber)
    {
        ostringstream oOStrStream;
        oOStrStream << pNumber;
        return oOStrStream.str();
    }

}

