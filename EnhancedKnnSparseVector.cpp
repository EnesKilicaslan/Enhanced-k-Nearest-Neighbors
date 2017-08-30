#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <set>

#include "EnhancedKnnSparseVector.h"
#include "Common.h"

using namespace std;

EnhancedKnnSparseVector::EnhancedKnnSparseVector(int k, const std::string &trainFileName, const std::string &testFileName,
                                                 bool pLabel)
        : EnhancedKNNBase(k,trainFileName, testFileName, pLabel) {
}

void EnhancedKnnSparseVector::fillVectors() {

    ifstream input(this->getTrainFileName().c_str());
    string line;
    getline( input, line ); // skip header

    while( getline( input, line ) ) {

        if(line == "\n")
            continue; //skip empty lines

        vector<string> labels_local;
        vector<bool> docs_local(words.size(), false);

        stringstream ss_main(line); // Turn the string into a stream.
        string tok_main;
        getline(ss_main, tok_main, ' ');
        string tok_lab;
        stringstream ss_lab(tok_main);

        while( getline(ss_lab, tok_lab, ',')){
            labels_local.push_back(tok_lab);
        }

        labels.push_back(labels_local); // append labels
        docs.push_back(docs_local);
        lengths.push_back(0); // first initialize lenght of the document to zero

        //parse word and occurance
        while(getline(ss_main, tok_main, ' ')){
            //cout <<"debug 2" << endl;

            unsigned long int pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance;
            stringstream ss( tok_main.substr(pos + 1, 1));
            ss >> occurance;
            int word_index = EKCommonOperations::contains(&words, word);

            if (word_index >= 0){
                //cout << "index Debug " << word << " " << word_index << "docs counter " << doc_counter <<  endl;
                //cout << "lend docs: " << docs.size() << endl;
                if(occurance > 0)
                    docs[this->getDocCounter()][word_index] = true;

                lengths[this->getDocCounter()] += occurance;
                this->setTotalLenOfDocs(this->getTotalLenOfDocs()+ occurance);

            } else {
                words.push_back(word);

                //!! Attention -- this makes really sparse matrix
                for( int i=0; i < docs.size(); ++i) //because this is new word, it never occurred before
                    docs[i].push_back(false);  // so make it 0 for the other documents

                docs[this->getDocCounter()][words.size() - 1] = true; // Bag of words (not binary)
                lengths[this->getDocCounter()] += occurance;
                this->setTotalLenOfDocs(this->getTotalLenOfDocs()+ occurance);
            }
        }

        //if(this->docCounter % 1000 == 0)
        //    cout << this->docCounter << endl;
        this->setDocCounter(getDocCounter() + 1);
    }

    setLa(getTotalLenOfDocs()/getDocCounter()); //calculate average document length once
}

/**
 * @ref https://stackoverflow.com/questions/15106102/how-to-use-c-stdostream-with-printf-like-formatting
 */
void EnhancedKnnSparseVector::saveVectors() const {
    //std::cout << std::putf("this is a number: %d\n",i);
    FILE *outFile;

    outFile = fopen(OUT_VEC_FILE_NAME, "w+");

    for(vector<string>::const_iterator it = words.begin(); it != words.end(); ++it) {

        fprintf(outFile, "%8s | ", (*it).c_str());
    }

    fprintf(outFile,"\n");
    std::vector< std::vector<bool> >::const_iterator row;
    std::vector<bool>::const_iterator col;

    for (row = this->docs.begin(); row != this->docs.end(); ++row)
    {
        for (col = row->begin(); col != row->end(); ++col)
            fprintf(outFile,"%8s | ", to_string(*col).c_str() );

        fprintf(outFile,"\n");
    }

}

double EnhancedKnnSparseVector::similarityBM25(std::vector<bool> const &s1, int len1, std::vector<bool> const &s2, int len2  ) const{

    double result=0.0;

    if(s1.size() != s2.size())
        return  -1; // sizes must be the same

    for(int i=0; i<s1.size(); ++i){
        if(s1[i] && s2[i]) {
            /*cout << "ei: " << i << endl;
            cout << "idf: " << idf(i) << endl;
            cout << "s1:  " << fPrime(i, s1) << endl;
            cout << "s2:  " << fPrime(i, s2) << endl;*/
            result += fPrime(s1, len1) *  fPrime(s2, len2) * idf(i);
        }
    }

    return result;
}

/**
 *
 *                 (k1 + 1) f(w,s)
 * f'(w,s) = -----------------------------
 *                                |s|
 *              k1 + ( 1 - b + b ----- )
 *                                la
 */
double EnhancedKnnSparseVector::fPrime(std::vector<bool> s, int len) const {
    // because we don't count the words in each document, we don't know f(w,s) which
    // is number of times that w occurs in document s
    // we just ignore it, I mean we take it 1 for each word in any document
    return ( (EKCommonOperations::K1 + 1)  /  (EKCommonOperations::K1 + ( 1 - EKCommonOperations::b + EKCommonOperations::b * len / getLa() )) );
}

/**
 * @ref: http://opensourceconnections.com/blog/2015/10/16/bm25-the-next-generation-of-lucene-relevation/
 * Trick: adding 1 before taking log, prevents us to get negeative result like said in referance
 *
 *
 *                    N - n(w) + 0.5
 *  idf(w) = log ( ------------------- + 1 )
 *                      n(w) + 0.5
**/
double EnhancedKnnSparseVector::idf(int wIndex) const {
    int nw = n(wIndex);
    double result = log10(((getDocCounter() - nw + 0.5) / (nw + 0.5)) + 1);

    /*cout << "*********" << endl;
    cout << "nw: "<< nw << " N: " << docCounter << endl;
    cout << "result: " << result << endl;
    cout << "up: " << (docCounter - nw + 0.5) << endl;
    cout << "down: " << (nw + 0.5) << endl;
    cout << "*********" << endl;
    */
    return result;
}

//this method takes directly index of the word, so equally  the column index of the documents
int EnhancedKnnSparseVector::n(int indx) const {
    int result = 0;

    //here again we don't have a notion about the counts of any word
    //we accept it like it occured once in a document
    for(int i=0; i< getDocCounter(); ++i) {
        bool occ = docs[i][indx];
        if (occ)
            result += 1;
    }

    return result;
}

// takes sparse vector that contains a notion for each word in the corpus
// So its size must be the same as words vector variable field
std::vector<std::string> EnhancedKnnSparseVector::enhancedKnn(int testDocumentIndex) const {
    std::vector< std::pair< int, double> > neighborSimPairs;
    set<string> candidateClasses;
    map<string , double> labelScores;
    vector<double> tresholds;
    vector<pair<string, double> > classScores; // this will be a copy of map lavelScores in order to traverse easyly
    vector<string> result; // the result labels
    int index_counter=0;


    //get the first k documents in order
    for(; index_counter<this->getK() && index_counter < getDocCounter(); ++index_counter)
        neighborSimPairs.push_back(std::pair<int, double> (index_counter,similarityBM25(testDocs[testDocumentIndex], testLengths[testDocumentIndex], docs[index_counter], lengths[index_counter])) ) ;

    //sort documents
    sort( neighborSimPairs.begin(), neighborSimPairs.end(), EKCommonOperations::neighborSimPairCompare);

    // traverse all the documents in the training set
    // and replace the new document if there is lower similar document
    // in the k nearest neighbors
    for(; index_counter< getDocCounter() ; ++index_counter){
        neighborSimPairs.push_back(std::pair<int, double> (index_counter,similarityBM25(testDocs[testDocumentIndex], testLengths[testDocumentIndex], docs[index_counter], lengths[index_counter])) ) ;
        sort( neighborSimPairs.begin(), neighborSimPairs.end(),  EKCommonOperations::neighborSimPairCompare);
        //remove last element
        //we always keep first k documents!
        neighborSimPairs.pop_back();
    }

    //user did not asked to calculate label
    //so nearest documents will be retrieved
    if(!isPLabel()) {
        vector<string> nearest_documents;
        for (int i = 0; i < neighborSimPairs.size(); ++i)
            nearest_documents.push_back(to_string(neighborSimPairs[i].first));

        return nearest_documents;
    }


    //cout << "Size of words in the corpus: " << words.size() << endl;
    //cout << "Neigbors: " << endl;

    //for (int m = 0; m < neighborSimPairs.size(); ++m) {
    //    cout << neighborSimPairs[m].first << " - " << neighborSimPairs[m].second << endl;
    //}

    /*
     * k nearest neigbors are found!
     * now applying weighted voting on candidate classes
     * accourding to the following formula
     *
     *                     ---`
     * score(c|se) =       \  y(si, c)BM25(si, se)^a
     *                     /__,
    */
    //put candidate classes in a set to make them unique
    for (int j = 0; j < neighborSimPairs.size(); ++j) {
        int labIndex = neighborSimPairs[j].first;

        for (int i = 0; i < labels[labIndex].size() ; ++i)
            candidateClasses.insert(labels[labIndex][i]);
    }

    //cout << "Candidate Classes: " << endl;
    //for(set<string>::const_iterator it= candidateClasses.begin(); it != candidateClasses.end(); ++it)
    //    cout << *it<< " ";

    //calculate score for each candidate class
    /**
     * for each candidate class
     *      for each document in the k-nearest neighbors
     *          calculate score
     */
    for(set<string>::const_iterator it= candidateClasses.begin(); it != candidateClasses.end(); ++it){
        labelScores[*it] = 0.0;

        for (int j = 0; j < neighborSimPairs.size(); ++j) {

            int labIndex = neighborSimPairs[j].first;

            //if the document contains candidate class
            if (std::find(labels[labIndex].begin(), labels[labIndex].end(), *it) != labels[labIndex].end()){
                double sim = neighborSimPairs[j].second;

                labelScores[*it] += pow(sim, getAlpha());
            }
        }
    }

    /**
     * Weighted voting is implemented above
     * scores for each candidate classes are stored in labelScores map (dictionary)
     *
     *
     * Distinctive Score-Cut Thresholding Strategy
     *
     *                  {  1   Si/Si >= Ti
     * Score-Cut(ci) = {
     *                  {  0   otherwise
     *
     *                  t1 is 1, because we always select top label
     */
    //initialize tresholds
    for(int i=0; i<labelScores.size(); ++i){
        tresholds.push_back(1-i*0.02);
    }

    //export the map to vector of pairs
    copy(labelScores.begin(),
         labelScores.end(),
         back_inserter<vector<pair<string, double > > > (classScores));

    //sort  classes vector in descending order with respect to scores of each class
    sort(classScores.begin(), classScores.end(), EKCommonOperations::pairCompareStr);

    //for (int i1 = 0; i1 < classScores.size(); ++i1) {
    //    cout << classScores[i1].first << " - " << classScores[i1].second << endl;
    //}
    //  cout << "result size: " << result.size() <<  endl;

    for (int l = 0; l <classScores.size() ; ++l)
        if(classScores[l].second / classScores[0].second >= tresholds[l] ) {
            /*cout << "label: " << classScores[l].first << "  score: "
                 << classScores[l].second << "  treshold: " << tresholds[l] << " ratio: "
                 << classScores[l].second/ classScores[0].second<< endl;*/
            result.push_back(classScores[l].first);
        }
        else
            break; // Note that δ(ci) is considered only if δ(c1),δ(c2),. . . ,δ(ci−1) all output 1.

    //cout << "result size: " << result.size() <<  endl;
    return result;
}

void EnhancedKnnSparseVector::runTest(){

    ofstream output;
    output.open("./eKNN-Result.txt");

    output << "labels" << endl;

    for(int i=0; i<testDocs.size(); ++i){
        vector<string> res = enhancedKnn(i);

        for (int j = 0; j < res.size() ; ++j)
            output << res[j] << " ";

        output << endl;
    }
}

void EnhancedKnnSparseVector::fillTestVectors(){
    ifstream input(getTestFileName().c_str());
    string line;
    getline( input, line ); // skip header
    int testDocCounter = 0;


    while( getline( input, line ) ) {
        //cout << line << endl;
        if (line == "\n")
            continue; //skip empty lines

        vector<bool > docs_local(words.size(), 0);
        testLengths.push_back(0);

        stringstream ss_main(line); // Turn the string into a stream.
        string tok_main;
        getline(ss_main, tok_main, ' ');
        string tok_lab;
        stringstream ss_lab(tok_main);

        testDocs.push_back(docs_local);

        //parse word and occurance
        while (getline(ss_main, tok_main, ' ')) {
            //cout <<"debug 2" << endl;

            unsigned long pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance;
            stringstream ss(tok_main.substr(pos + 1, 1));
            ss >> occurance;
            int word_index = EKCommonOperations::contains(&words, word);

            if (word_index == -1)
                continue; //this word has never seen before so, it does not have any effect
            else {
                testDocs[testDocCounter][word_index] = true;
                testLengths[testDocCounter] += occurance;
            }

        }

        testDocCounter += 1;
    }

    /*
    cout << "***test vectors: " << endl;
    std::vector< std::vector<bool > >::const_iterator row;
    std::vector<bool >::const_iterator col;
    cout << "size of tests: " << testDocs.size();
    for (row = this->testDocs.begin(); row != this->testDocs.end(); ++row)
    {
        for (col = row->begin(); col != row->end(); ++col)
        {
            printf("%8s | ", EnesKilicaslanCommonOperations::numberToString(*col).c_str());
        }
        cout << endl;
    }*/

}


void EnhancedKnnSparseVector::run(bool save) {

    cout << "Started filling train vectors.." << endl;
    this->fillVectors();
    cout << "Filling train vectors is done!" << endl << endl;

    if(save) {
        cout << "Saving vectors to file named '" << OUT_VEC_FILE_NAME << "' " << endl;
        this->saveVectors();
        cout << "Vectors are saved! you can check the file " << OUT_VEC_FILE_NAME << endl << endl;
    }

    cout << "Started filling test vectors.." << endl;
    this->fillTestVectors();
    cout << "Filling test vectors is done!" << endl << endl;

    (isPLabel()) ? cout << "Making prediction.." << endl : cout << "Calculating nearest neighbours.." << endl ;
    this->runTest();

    //** get time passed from the begining of the application
    //** and print it in minutes
    cout << "Operation Completed!" << endl;

}