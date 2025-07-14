#pragma once

#include <cmath>
#include <array>

class SimplexNoise {
private:
    static const float F2;
    static const float F3;
    static const float F4;
    static const float G2;
    static const float G3;
    static const float G4;

    static const int p[256];
    static int perm[512];
    static int permMod12[512];

    static const float grad3[12][3];
    static const float grad4[32][4];

    static float dot(const float g[3], float x, float y, float z);
    static float dot(const float g[4], float x, float y, float z, float w);
    static int fastfloor(float x);

public:
    static void initPerm();

    static float noise(float x, float y);
    static float noise(float x, float y, float z);
    static float noise(float x, float y, float z, float w);
    static float noise(float x, float y, float z, int dimensions);
    static int get_p(int idx);
    static int get_perm(int idx);
    static int get_permMod12(int idx);
    static void get_grad3(int idx, float grad[3]);
    static float dot_product(float* g, float x, float y, float z);
    static float simplex_noise(float x, float y, float z = 0.0f, int dimensions = 3);
};