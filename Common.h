//
// Created by Enes Kılıçaslan on 26/08/17.
//

#ifndef INVERTEDINDEX1_COMMON_H
#define INVERTEDINDEX1_COMMON_H


#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>


namespace EKCommonOperations {

    // const values for fprime method
    const double K1 = 6.0;
    const double b = 0.9;


    int lenOfDoc(std::vector<std::pair<long, int> > const &s){
        int res = 0;

        std::vector<std::pair<long, int> >::const_iterator it;

        for(it=s.begin(); it != s.end(); ++it)
            res += it->second;

        return res;
    }

    bool wordOccPairCompare(const std::pair<long, int> &firstElem , const std::pair<long, int> secondElem) {
        return firstElem.first < secondElem.first;
    }

    //@ref: https://stackoverflow.com/questions/19483663/vector-intersection-in-c
    std::vector<long> instersectionWords( std::vector<std::pair<long, int> > const &v1, std::vector<std::pair<long, int> > const &v2) {

        std::vector<long> v3;
        ///we dont need to sort here because they are already sorted
        //sort(v1.begin(), v1.end());
        //sort(v2.begin(), v2.end());
        set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v3), wordOccPairCompare);

        return v3;
    }

}




#endif //INVERTEDINDEX1_COMMON_H
