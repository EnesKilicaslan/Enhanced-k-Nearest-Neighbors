#include "EnhancedKNNBase.h"

EnhancedKNNBase::EnhancedKNNBase(int k, const std::string &trainFileName, const std::string &testFileName, bool pLabel)
        : trainFileName(trainFileName), testFileName(testFileName), pLabel(pLabel)  {

    this->k = k;
    this->alpha = k/10;

    this->docCounter = 0;
    this->N =0;
    this->totalLenOfDocs =0;

}

const std::string &EnhancedKNNBase::getTrainFileName() const {
    return trainFileName;
}

void EnhancedKNNBase::setTrainFileName(const std::string &trainFileName) {
    EnhancedKNNBase::trainFileName = trainFileName;
}

const std::string &EnhancedKNNBase::getTestFileName() const {
    return testFileName;
}

void EnhancedKNNBase::setTestFileName(const std::string &testFileName) {
    EnhancedKNNBase::testFileName = testFileName;
}

int EnhancedKNNBase::getK() const {
    return k;
}

void EnhancedKNNBase::setK(int k) {
    EnhancedKNNBase::k = k;
}

double EnhancedKNNBase::getAlpha() const {
    return alpha;
}

void EnhancedKNNBase::setAlpha(double alpha) {
    EnhancedKNNBase::alpha = alpha;
}

long EnhancedKNNBase::getN() const {
    return N;
}

void EnhancedKNNBase::setN(long N) {
    EnhancedKNNBase::N = N;
}

long EnhancedKNNBase::getDocCounter() const {
    return docCounter;
}

void EnhancedKNNBase::setDocCounter(long docCounter) {
    EnhancedKNNBase::docCounter = docCounter;
}

long EnhancedKNNBase::getTotalLenOfDocs() const {
    return totalLenOfDocs;
}

void EnhancedKNNBase::setTotalLenOfDocs(long totalLenOfDocs) {
    EnhancedKNNBase::totalLenOfDocs = totalLenOfDocs;
}

double EnhancedKNNBase::getLa() const {
    return la;
}

void EnhancedKNNBase::setLa(double la) {
    EnhancedKNNBase::la = la;
}

bool EnhancedKNNBase::isPLabel() const {
    return pLabel;
}

void EnhancedKNNBase::setPLabel(bool pLabel) {
    EnhancedKNNBase::pLabel = pLabel;
}
