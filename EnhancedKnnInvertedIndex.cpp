#include <math.h>
#include <fstream>
#include <sstream>
#include <set>

#include "EnhancedKnnInvertedIndex.h"

#include "Common.h"

using namespace std;

EnhancedKnnInvertedIndex::EnhancedKnnInvertedIndex(int k, const std::string &trainFileName, const std::string &testFileName,
                                                 bool pLabel)
        : EnhancedKNNBase(k,trainFileName, testFileName, pLabel) {
}


void EnhancedKnnInvertedIndex::fillInvertedIndexes() {
    ifstream input(getTrainFileName().c_str());
    string line;
    getline( input, line ); // skip header

    while( getline( input, line ) ) {
        long docLen = 0;

        vector< pair <long, int> > local_doc;

        if(line == "\n")
            continue; //skip empty lines

        vector<string> labels_local;

        stringstream ss_main(line); // Turn the string into a stream.
        string tok_main;
        getline(ss_main, tok_main, ' ');
        string tok_lab;
        stringstream ss_lab(tok_main);

        while( getline(ss_lab, tok_lab, ',')){
            labels_local.push_back(tok_lab);
        }

        labels.push_back(labels_local); // append labels

        while(getline(ss_main, tok_main, ' ')) {

            unsigned long int pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance;
            stringstream ss( tok_main.substr(pos + 1, 1));
            ss >> occurance;
            //cout << word << ":" << occurance << "-";

            if(occurance > 0){
                //filling index1 which maps doc to (word, frequency) pairs
                std::string::size_type sz;
                long word_dec = std::stol (word,&sz);
                local_doc.push_back(pair<long,int> (word_dec, occurance));

                //filling index2 which maps word to a set of documents
                this->wordToDocs[word_dec].push_back(getDocCounter());
            }

            docLen += occurance;
        }
        //keeping the words in inverted index gains us more efficiency
        std::sort(local_doc.begin(), local_doc.end(), EKCommonOperations::wordOccPairCompare);

        docToWords.push_back(local_doc);
        this->setTotalLenOfDocs(getTotalLenOfDocs() + docLen);
        this->setDocCounter(getDocCounter() + 1);

    }

    setN(docToWords.size());
    calculateLa();
}

void EnhancedKnnInvertedIndex::saveInvertedIndex() const {

    ofstream outFile;
    outFile.open(OUT_INVERTED_INDEX_FILE_NAME);

    std::vector< std::vector <pair<long, int> > >::const_iterator it;

    outFile << "doc_id -> word1 , word2 , word3 , ..." << endl;
    for(it=docToWords.begin(); it!=docToWords.end(); ++it){

        outFile << it - docToWords.begin() << " => ";
        vector<pair<long, int> >::const_iterator i;

        for (i=it->begin(); i != it->end(); ++i) {
            outFile << i->first << " ";
        }

        outFile << endl;
    }

    outFile << endl;
    outFile << "*****************************************************" << endl << endl;

    map< long , std::vector<long > >::const_iterator iter;

    outFile << "word_id -> doc1 , doc2 , doc3 , ..." << endl;
    for(iter= wordToDocs.begin(); iter != wordToDocs.end(); ++iter){
        outFile << iter->first << " => ";

        vector<long>::const_iterator i;

        for ( i = iter->second.begin(); i < iter->second.end() ; ++i) {
            outFile << *i << " ";
        }

        outFile << endl;
    }

}

void EnhancedKnnInvertedIndex::fillTestInvertedIndex() {
    ifstream input(getTestFileName().c_str());
    string line;
    getline( input, line ); // skip header

    while( getline( input, line ) ) {
        long docLen = 0;

        if (line == "\n")
            continue; //skip empty lines

        vector< pair <long, int> > local_doc;

        stringstream ss_main(line); // Turn the string into a stream.
        string tok_main;
        getline(ss_main, tok_main, ' ');
        string tok_lab;
        stringstream ss_lab(tok_main);

        //parse word and occurance
        while (getline(ss_main, tok_main, ' ')) {
            //cout <<"debug 2" << endl;

            unsigned long pos = tok_main.find(':');
            string word = tok_main.substr(0, pos);
            int occurance;
            stringstream ss(tok_main.substr(pos + 1, 1));
            ss >> occurance;

            if(occurance > 0){
                std::string::size_type sz;
                long word_dec = std::stol (word,&sz);
                local_doc.push_back(pair<long,int> (word_dec, occurance));
            }

            docLen += occurance;
        }

        //keeping the words in inverted index gains us more efficiency
        std::sort(local_doc.begin(), local_doc.end(), EKCommonOperations::wordOccPairCompare);

        testDocToWords.push_back(local_doc);
    }
}

int EnhancedKnnInvertedIndex::n(long w) const {
    return wordToDocs.at(w).size();
}

double EnhancedKnnInvertedIndex::idf(long w) const {
    return log((getN() - n(w) + 0.5) / n(w) + 0.5);
}

/**
 * returns term frequency of w in document s
 * @param w
 * @param s
 * @return
 */
int EnhancedKnnInvertedIndex::f(long w, vector<std::pair<long, int> > const &s) const {
    vector<std::pair<long, int> >::const_iterator it;

    for(it = s.begin(); it!=s.end(); ++it)
        if (it->first == w)
            return it->second;
    return 0;
}

double EnhancedKnnInvertedIndex::fPrime(long w, vector<std::pair<long, int> > const &s) const {

    return ( ((EKCommonOperations::K1 + 1) * f(w,s) ) /
             (EKCommonOperations::K1 + ( 1 - EKCommonOperations::b + EKCommonOperations::b * EKCommonOperations::lenOfDoc(s) / getLa() )));
}

double EnhancedKnnInvertedIndex::similarityBM25(std::vector< pair<long, int> > const &s1, std::vector< pair<long, int> > const &s2) const{
    double result= 0.0;

    vector<long> words = EKCommonOperations::instersectionWords(s1, s2);
    //cout << "intersection: ";

    for(int i=0; i<words.size(); ++i) {
        //cout << "(" << words[i].first << "," << words[i].second << ") ";
        result += fPrime(words[i], s1) * fPrime(words[i], s2) * idf(words[i]);
    }
    //cout << endl;
    //cout << "result: " << result << endl;

    return result;
}

std::vector<std::string> EnhancedKnnInvertedIndex::enhancedKnn(int testDocumentIndex) const {

    /**
     * For a test document ->
     * get all words from index1,
     * then for each word of test document find all training documents from index2
     * and do union of those sets
     * == all docs sharing at least 1 word with test document. Then compute similarities
    */
    set<string> candidateClasses;
    map<string , double> labelScores;
    vector<double> tresholds;
    vector<pair<string, double> > classScores; // this will be a copy of map lavelScores in order to traverse easyly
    vector<string> result; // the result labels


    set< long > docsHaveCommonWords;    //Indexes of documents which have at least one common word with the test document
    vector< std::pair< long , double > > neighborSimPairs; //at the end, this data structure keeps k nearest neighbors
    vector< long > wordsInTestDoc = EKCommonOperations::getWordsOfDocument(testDocToWords[testDocumentIndex]);

    double count_index = 1;
    int count_iter=0;

    //cout << "how many words: " << wordsInTestDoc.size() << endl;

    /**
     *      set: docs_have_common_words
     *
     *      for each w in test documents:
     *          for each s in wordToDocs[w]:
     *              docs_have_common_words.append(s)
     */
    while(docsHaveCommonWords.size() == 0 && count_iter < EKCommonOperations::MAX_ITERATION) { //the condition in while loop is to make sure a to find similar docs for a test document
        for(vector< long >::const_iterator it= wordsInTestDoc.begin(); it != wordsInTestDoc.end(); ++it) {
            try {
                //cout << wordToDocs.at(*it).size() << " number of documents contain word " << *it << endl;

                //lots of documents contain the corresponding word, so skip it
                // but needs to put extra direct restriction (threshold) for moderate size of input
                if(wordToDocs.at(*it).size() > this->getN() / EKCommonOperations::MAX_N_FOR_RATIO && wordToDocs.at(*it).size() > EKCommonOperations::MAX_N_FOR_DIRECT * count_index)// && wordToDocs.at(*it).size() > 100000 )
                    continue;

                for (vector<long>::const_iterator i = wordToDocs.at(*it).begin(); i != wordToDocs.at(*it).end(); ++i){
                    docsHaveCommonWords.insert(*i);
                }
            }
            catch (const std::out_of_range& oor) {
                //the word in the test document never appeared in any of training documents
                //cout << "word " << *it << " does not exist in train" << endl;
                continue;
            }
        }

        count_index += 1.5; //increase treshold until find documents which have common word with the test document
        ++count_iter;
    }

    //cout << "first word size: " << wordToDocs.at(wordsInTestDoc[0]).size() << endl;
    //cout << "******how many docs: " << docsHaveCommonWords.size() << endl;

    //compute similarities for each document that has at least one common word
    for(set<long>::const_iterator it = docsHaveCommonWords.begin(); it != docsHaveCommonWords.end(); ++it) {
        //cout <<"calcluating sim for doc " << *it << endl;


        neighborSimPairs.push_back(std::pair<long, double> (*it, similarityBM25(docToWords[*it], testDocToWords[testDocumentIndex])));
        sort( neighborSimPairs.begin(), neighborSimPairs.end(), EKCommonOperations::neighborSimPairCompare);
    }

    //keep only k nearest neighbors, so delete the others
    //if(neighborSimPairs.size() > k)
    //    for(int i=k; i<neighborSimPairs.size(); ++i)
    //        neighborSimPairs.pop_back();

    //cout << "knn calculated for test document " << testDocumentIndex  << endl;

    //*************** burada asagisini yapistirdim: TODO: sil bu satiiri
    //the rest is the same with the others
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
    for (int j = 0; j < neighborSimPairs.size() && j < getK(); ++j) {
        long labIndex = neighborSimPairs[j].first;

        for (int i = 0; i < labels[labIndex].size() ; ++i)
            candidateClasses.insert(labels[labIndex][i]);
    }

    //cout << "candidate class size: " << candidateClasses.size() << endl;
    //*************** burada asagisini yapistirdim: TODO: sil bu satiiri
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

            long labIndex = neighborSimPairs[j].first;

            //if the document contains candidate class
            if (std::find(labels[labIndex].begin(), labels[labIndex].end(), *it) != labels[labIndex].end()){
                double sim = neighborSimPairs[j].second;

                labelScores[*it] += pow(sim,getAlpha());
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
        tresholds.push_back(EKCommonOperations::getThreshold(i));
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
    //cout << "class scores size: " << classScores.size() << endl;

    for (int l = 0; l <classScores.size() ; ++l){
        // cout << "class: " <<classScores[l].first << "\tscore: " << classScores[l].second << endl;
        if(classScores[l].second / classScores[0].second >= tresholds[l] ) {
            /*cout << "label: " << classScores[l].first << "  score: "
                 << classScores[l].second << "  treshold: " << tresholds[l] << " ratio: "
                 << classScores[l].second/ classScores[0].second<< endl;*/
            result.push_back(classScores[l].first);
        }
        else
            break; // Note that δ(ci) is considered only if δ(c1),δ(c2),. . . ,δ(ci−1) all output 1.

    }

    //cout << "press any key to continue next document: ";
    //string nothing;
    //cin >> nothing;
    //cout << "result size: " << result.size() <<  endl;
    return result;
}

void EnhancedKnnInvertedIndex::runTest(){

    ofstream output;
    output.open("./eKNN-Result.txt");

    clock_t t;
    t = clock();

    output << "labels" << endl;

    for(long i=0; i<testDocToWords.size(); ++i){

        if(i % 10000 == 0 && i != 0){
            t = clock() - t;
            printf ("%ld documents done in %ld minutes!\n",i, (((long)t)/CLOCKS_PER_SEC)/60);
            t = clock();
        }

        vector<string> res = enhancedKnn(i);

        for (int j = 0; j < res.size() ; ++j)
            output << res[j] << " ";

        output << endl;
    }
}

//before using this method, N must be set!
void EnhancedKnnInvertedIndex::calculateLa(){
    this->setLa(this->getTotalLenOfDocs() / (double) getN());
}

void EnhancedKnnInvertedIndex::run(bool save) {

    cout << "Started filling train inverted index.." << endl;
    this->fillInvertedIndexes();
    cout << "Filling inverted index is done!" << endl << endl;

    if(save) {
        cout << "Saving inverted indexes to file named '" << OUT_INVERTED_INDEX_FILE_NAME << "' " << endl;
        this->saveInvertedIndex();
        cout << "inverted indexes are saved! you can check the file " << OUT_INVERTED_INDEX_FILE_NAME << endl << endl;
    }

    cout << "Started filling test inverted index.." << endl;
    this->fillTestInvertedIndex();
    cout << "Filling test inverted index is done!" << endl << endl;

    printInformation();

    (isPLabel()) ? cout << "Making prediction.." << endl : cout << "Calculating nearest neighbours.." << endl ;
    this->runTest();

}

void EnhancedKnnInvertedIndex::printInformation() const{
    //prints total number, total length of training documents and average of them
    cout << "************ information about training documents ***************" << endl;
    cout << "N: " << getN() << endl;
    cout << "la: " << getLa() << endl;
    cout << "total len of docs: " << getTotalLenOfDocs()<< endl;
    cout << "*****************************************************************" << endl << endl;
}