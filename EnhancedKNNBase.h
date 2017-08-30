
#ifndef EKNN4_ENHANCEDKNNINTERFACE_H
#define EKNN4_ENHANCEDKNNINTERFACE_H

#include <iostream>

class EnhancedKNNBase {

public:
    EnhancedKNNBase(int k, const std::string &trainFileName, const std::string &testFileName, bool pLabel);

    //getter and setters
    const std::string &getTrainFileName() const;
    void setTrainFileName(const std::string &trainFileName);

    const std::string &getTestFileName() const;
    void setTestFileName(const std::string &testFileName);

    int getK() const;
    void setK(int k);

    double getAlpha() const;
    void setAlpha(double alpha);

    long getN() const;
    void setN(long N);

    long getDocCounter() const;
    void setDocCounter(long docCounter);

    long getTotalLenOfDocs() const;
    void setTotalLenOfDocs(long totalLenOfDocs);

    double getLa() const;
    void setLa(double la);

    bool isPLabel() const;
    void setPLabel(bool pLabel);

private:

    std::string trainFileName;
    std::string testFileName;

    int k; // this is the 'famous' k
    double alpha;

    long N; //number of documents
    long docCounter, totalLenOfDocs; //total lenght of training documents
    double la;  //average lenght of training documents

    bool pLabel;
};


#endif //EKNN4_ENHANCEDKNNINTERFACE_H
