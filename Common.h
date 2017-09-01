
#ifndef COMMON_H
#define COMMON_H


#include <iostream>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

using namespace std;

namespace EKCommonOperations
{
    //const values for fprime method
    //constants for BM25 similarity
    const static double K1 = 6.0;
    const static double b = 0.9;
    const static double THRESHOLD_CONST = 10.0;

    //these two values are selected with respect to our time limit which is 6 hours for WikiLSHTC dataset
    const static long MAX_N_FOR_RATIO = 11000;
    const static long MAX_N_FOR_DIRECT = 160;
    const static int MAX_ITERATION = 30;

    int static lenOfDoc(std::vector<std::pair<long, int> > const &s){
        int res = 0;
        std::vector<std::pair<long, int> >::const_iterator it;

        for(it=s.begin(); it != s.end(); ++it)
            res += it->second;

        return res;
    }

    double static getThreshold(int indx) {
        if(indx == 0)
            return 1.0;
        else if( indx == 1)
            return 0.3;
        else
            return (double) indx / THRESHOLD_CONST;
    }


    bool static wordOccPairCompare(const std::pair<long, int> &firstElem, const std::pair<long, int> secondElem) {
        return firstElem.first < secondElem.first;
    }

    std::vector<long > static getWordsOfDocument(std::vector<std::pair<long, int> > const &s) {
        std::vector<std::pair<long, int> >::const_iterator it;
        std::vector<long > res;

        for(it=s.begin(); it!= s.end(); ++it)
            res.push_back(it->first);

        return res;
    }

    //@ref: https://stackoverflow.com/questions/19483663/vector-intersection-in-c
    std::vector<long> static instersectionWords( std::vector<std::pair<long, int> > const &v1, std::vector<std::pair<long, int> > const &v2) {

        std::vector<std::pair<long, int> > v3;
        std::vector<long> res;

        ///we dont need to sort here because they are already sorted
        //sort(v1.begin(), v1.end());
        //sort(v2.begin(), v2.end());
        set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v3), wordOccPairCompare);

        for(std::vector<std::pair<long, int> >::const_iterator it= v3.begin(); it != v3.end(); ++it)
            res.push_back(it->first);

        return res;
    }

    bool static neighborSimPairCompare(const std::pair<int, double >& firstElem,
                                       const std::pair<int, double >& secondElem) {
        return firstElem.second > secondElem.second;
    }

    bool static pairCompareStr(const std::pair<std::string, double >& firstElem,
                               const std::pair<std::string, double >& secondElem) {
        return firstElem.second > secondElem.second;
    }

    template<typename T, typename A>
    int static contains(vector<T, A> const *v, T x) {
        typename vector<T, A>::const_iterator it = find(v->begin(), v->end(), x); //iterator

        if (it == v->end())
            return -1;
        else
            return it - v->begin();
    }

    int static isDirectory(const char *path) {
        struct stat statbuf;
        if (stat(path, &statbuf) != 0)
            return 0;
        return S_ISDIR(statbuf.st_mode);
    }


}


#endif //COMMON_H
