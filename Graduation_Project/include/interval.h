#ifndef INTERVAL_H
#define INTERVAL_H

class interval {
  public:
    double minv, maxv;

    interval() : minv(+infinity), maxv(-infinity) {} // Default interval is empty

    interval(double _min, double _max) : minv(_min), maxv(_max) {}

    bool contains(double x) const {
        return minv <= x && x <= maxv;
    }

    bool surrounds(double x) const {
        return minv < x && x < maxv;
    }

    double clamp(double x) const {
        if (x < minv) return minv;
        if (x > maxv) return maxv;
        return x;
    }
    static const interval empty, universe;
};

const static interval empty   (+infinity, -infinity);
const static interval universe(-infinity, +infinity);


#endif