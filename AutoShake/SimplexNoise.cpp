#include "SimplexNoise.h"
#include <iostream>
#include <iomanip>

const float SimplexNoise::F2 = 0.366025404f;
const float SimplexNoise::F3 = 0.333333333f;
const float SimplexNoise::F4 = (sqrtf(5.0f) - 1.0f) / 4.0f;
const float SimplexNoise::G2 = 0.211324865f;
const float SimplexNoise::G3 = 0.166666667f;
const float SimplexNoise::G4 = (5.0f - sqrtf(5.0f)) / 20.0f;

const float SimplexNoise::grad3[12][3] = {
    {1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f},
    {1.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, -1.0f},
    {0.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 1.0f}, {0.0f, 1.0f, -1.0f}, {0.0f, -1.0f, -1.0f}
};

const float SimplexNoise::grad4[32][4] = {
    {0.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f, -1.0f}, {0.0f, 1.0f, -1.0f, 1.0f}, {0.0f, 1.0f, -1.0f, -1.0f},
    {0.0f, -1.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 1.0f, -1.0f}, {0.0f, -1.0f, -1.0f, 1.0f}, {0.0f, -1.0f, -1.0f, -1.0f},
    {1.0f, 0.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 1.0f, -1.0f}, {1.0f, 0.0f, -1.0f, 1.0f}, {1.0f, 0.0f, -1.0f, -1.0f},
    {-1.0f, 0.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 1.0f, -1.0f}, {-1.0f, 0.0f, -1.0f, 1.0f}, {-1.0f, 0.0f, -1.0f, -1.0f},
    {1.0f, 1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f, -1.0f}, {1.0f, -1.0f, 0.0f, 1.0f}, {1.0f, -1.0f, 0.0f, -1.0f},
    {-1.0f, 1.0f, 0.0f, 1.0f}, {-1.0f, 1.0f, 0.0f, -1.0f}, {-1.0f, -1.0f, 0.0f, 1.0f}, {-1.0f, -1.0f, 0.0f, -1.0f},
    {1.0f, 1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, -1.0f, 0.0f},
    {-1.0f, 1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 1.0f, 0.0f}, {-1.0f, -1.0f, -1.0f, 0.0f}
};

const int SimplexNoise::p[256] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
};

int SimplexNoise::perm[512];
int SimplexNoise::permMod12[512];

void SimplexNoise::initPerm() {
    for (int i = 0; i < 512; i++) {
        perm[i] = p[i % 256];
        permMod12[i] = perm[i] % 12;
    }
}

int SimplexNoise::get_p(int idx) {
    return p[idx & 0xFF];
}

int SimplexNoise::get_perm(int idx) {
    return p[idx & 0xFF];
}

int SimplexNoise::get_permMod12(int idx) {
    return get_perm(idx) % 12;
}

void SimplexNoise::get_grad3(int idx, float grad[3]) {
    idx = idx % 12;
    grad[0] = grad3[idx][0];
    grad[1] = grad3[idx][1];
    grad[2] = grad3[idx][2];
}

int SimplexNoise::fastfloor(float x) {
    int xi = (int)x;
    return x < xi ? xi - 1 : xi;
}

float SimplexNoise::dot_product(float* g, float x, float y, float z) {
    return g[0] * x + g[1] * y + g[2] * z;
}

float SimplexNoise::dot(const float g[3], float x, float y, float z) {
    return g[0] * x + g[1] * y + g[2] * z;
}

float SimplexNoise::dot(const float g[4], float x, float y, float z, float w) {
    return g[0] * x + g[1] * y + g[2] * z + g[3] * w;
}

float SimplexNoise::simplex_noise(float x, float y, float z, int dimensions) {
    if (dimensions == 2) {
        float n0, n1, n2;

        float s = (x + y) * F2;
        int i = fastfloor(x + s);
        int j = fastfloor(y + s);

        float t = (i + j) * G2;
        float X0 = i - t;
        float Y0 = j - t;
        float x0 = x - X0;
        float y0 = y - Y0;

        int i1, j1;
        if (x0 > y0) {
            i1 = 1;
            j1 = 0;
        }
        else {
            i1 = 0;
            j1 = 1;
        }

        float x1 = x0 - i1 + G2;
        float y1 = y0 - j1 + G2;
        float x2 = x0 - 1.0f + 2.0f * G2;
        float y2 = y0 - 1.0f + 2.0f * G2;

        int ii = i & 255;
        int jj = j & 255;
        int gi0 = get_permMod12(ii + get_perm(jj));
        int gi1 = get_permMod12(ii + i1 + get_perm(jj + j1));
        int gi2 = get_permMod12(ii + 1 + get_perm(jj + 1));

        float t0 = 0.5f - x0 * x0 - y0 * y0;
        if (t0 < 0) {
            n0 = 0.0f;
        }
        else {
            t0 *= t0;
            float g0[3];
            get_grad3(gi0, g0);
            n0 = t0 * t0 * dot_product(g0, x0, y0, 0);
        }

        float t1 = 0.5f - x1 * x1 - y1 * y1;
        if (t1 < 0) {
            n1 = 0.0f;
        }
        else {
            t1 *= t1;
            float g1[3];
            get_grad3(gi1, g1);
            n1 = t1 * t1 * dot_product(g1, x1, y1, 0);
        }

        float t2 = 0.5f - x2 * x2 - y2 * y2;
        if (t2 < 0) {
            n2 = 0.0f;
        }
        else {
            t2 *= t2;
            float g2[3];
            get_grad3(gi2, g2);
            n2 = t2 * t2 * dot_product(g2, x2, y2, 0);
        }

        return 70.0f * (n0 + n1 + n2);
    }
    else {
        float n0, n1, n2, n3;

        float s = (x + y + z) * F3;
        int i = fastfloor(x + s);
        int j = fastfloor(y + s);
        int k = fastfloor(z + s);

        float t = (i + j + k) * G3;
        float X0 = i - t;
        float Y0 = j - t;
        float Z0 = k - t;
        float x0 = x - X0;
        float y0 = y - Y0;
        float z0 = z - Z0;

        int i1, j1, k1;
        int i2, j2, k2;
        if (x0 >= y0) {
            if (y0 >= z0) {
                i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
            }
            else if (x0 >= z0) {
                i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
            }
            else {
                i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
            }
        }
        else {
            if (y0 < z0) {
                i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
            }
            else if (x0 < z0) {
                i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
            }
            else {
                i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
            }
        }

        float x1 = x0 - i1 + G3;
        float y1 = y0 - j1 + G3;
        float z1 = z0 - k1 + G3;
        float x2 = x0 - i2 + 2.0f * G3;
        float y2 = y0 - j2 + 2.0f * G3;
        float z2 = z0 - k2 + 2.0f * G3;
        float x3 = x0 - 1.0f + 3.0f * G3;
        float y3 = y0 - 1.0f + 3.0f * G3;
        float z3 = z0 - 1.0f + 3.0f * G3;

        int ii = i & 255;
        int jj = j & 255;
        int kk = k & 255;
        int gi0 = get_permMod12(ii + get_perm(jj + get_perm(kk)));
        int gi1 = get_permMod12(ii + i1 + get_perm(jj + j1 + get_perm(kk + k1)));
        int gi2 = get_permMod12(ii + i2 + get_perm(jj + j2 + get_perm(kk + k2)));
        int gi3 = get_permMod12(ii + 1 + get_perm(jj + 1 + get_perm(kk + 1)));

        float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
        if (t0 < 0) n0 = 0.0f;
        else {
            t0 *= t0;
            float g0[3];
            get_grad3(gi0, g0);
            n0 = t0 * t0 * dot_product(g0, x0, y0, z0);
        }

        float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
        if (t1 < 0) n1 = 0.0f;
        else {
            t1 *= t1;
            float g1[3];
            get_grad3(gi1, g1);
            n1 = t1 * t1 * dot_product(g1, x1, y1, z1);
        }

        float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
        if (t2 < 0) n2 = 0.0f;
        else {
            t2 *= t2;
            float g2[3];
            get_grad3(gi2, g2);
            n2 = t2 * t2 * dot_product(g2, x2, y2, z2);
        }

        float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
        if (t3 < 0) n3 = 0.0f;
        else {
            t3 *= t3;
            float g3[3];
            get_grad3(gi3, g3);
            n3 = t3 * t3 * dot_product(g3, x3, y3, z3);
        }

        return 32.0f * (n0 + n1 + n2 + n3);
    }
}

float SimplexNoise::noise(float x, float y, float z, int dimensions) {
    return simplex_noise(x, y, z, dimensions);
}

float SimplexNoise::noise(float x, float y) {
    return simplex_noise(x, y, 0.0f, 2);
}

float SimplexNoise::noise(float x, float y, float z) {
    return simplex_noise(x, y, z, 3);
}

float SimplexNoise::noise(float x, float y, float z, float w) {
    float n0, n1, n2, n3, n4;

    float s = (x + y + z + w) * F4;
    int i = fastfloor(x + s);
    int j = fastfloor(y + s);
    int k = fastfloor(z + s);
    int l = fastfloor(w + s);
    float t = (i + j + k + l) * G4;
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;
    float W0 = l - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;
    float w0 = w - W0;

    int rankx = 0;
    int ranky = 0;
    int rankz = 0;
    int rankw = 0;
    if (x0 > y0) rankx++; else ranky++;
    if (x0 > z0) rankx++; else rankz++;
    if (x0 > w0) rankx++; else rankw++;
    if (y0 > z0) ranky++; else rankz++;
    if (y0 > w0) ranky++; else rankw++;
    if (z0 > w0) rankz++; else rankw++;

    int i1 = rankx >= 3 ? 1 : 0;
    int j1 = ranky >= 3 ? 1 : 0;
    int k1 = rankz >= 3 ? 1 : 0;
    int l1 = rankw >= 3 ? 1 : 0;

    int i2 = rankx >= 2 ? 1 : 0;
    int j2 = ranky >= 2 ? 1 : 0;
    int k2 = rankz >= 2 ? 1 : 0;
    int l2 = rankw >= 2 ? 1 : 0;

    int i3 = rankx >= 1 ? 1 : 0;
    int j3 = ranky >= 1 ? 1 : 0;
    int k3 = rankz >= 1 ? 1 : 0;
    int l3 = rankw >= 1 ? 1 : 0;

    float x1 = x0 - i1 + G4;
    float y1 = y0 - j1 + G4;
    float z1 = z0 - k1 + G4;
    float w1 = w0 - l1 + G4;
    float x2 = x0 - i2 + 2.0f * G4;
    float y2 = y0 - j2 + 2.0f * G4;
    float z2 = z0 - k2 + 2.0f * G4;
    float w2 = w0 - l2 + 2.0f * G4;
    float x3 = x0 - i3 + 3.0f * G4;
    float y3 = y0 - j3 + 3.0f * G4;
    float z3 = z0 - k3 + 3.0f * G4;
    float w3 = w0 - l3 + 3.0f * G4;
    float x4 = x0 - 1.0f + 4.0f * G4;
    float y4 = y0 - 1.0f + 4.0f * G4;
    float z4 = z0 - 1.0f + 4.0f * G4;
    float w4 = w0 - 1.0f + 4.0f * G4;

    int ii = i & 255;
    int jj = j & 255;
    int kk = k & 255;
    int ll = l & 255;
    int gi0 = perm[perm[perm[perm[ll] + kk] + jj] + ii] % 32;
    int gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1 + perm[ll + l1]]]] % 32;
    int gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2 + perm[ll + l2]]]] % 32;
    int gi3 = perm[ii + i3 + perm[jj + j3 + perm[kk + k3 + perm[ll + l3]]]] % 32;
    int gi4 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1 + perm[ll + 1]]]] % 32;

    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0;
    if (t0 < 0) {
        n0 = 0.0f;
    }
    else {
        t0 *= t0;
        n0 = t0 * t0 * dot(grad4[gi0], x0, y0, z0, w0);
    }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1;
    if (t1 < 0) {
        n1 = 0.0f;
    }
    else {
        t1 *= t1;
        n1 = t1 * t1 * dot(grad4[gi1], x1, y1, z1, w1);
    }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2;
    if (t2 < 0) {
        n2 = 0.0f;
    }
    else {
        t2 *= t2;
        n2 = t2 * t2 * dot(grad4[gi2], x2, y2, z2, w2);
    }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3;
    if (t3 < 0) {
        n3 = 0.0f;
    }
    else {
        t3 *= t3;
        n3 = t3 * t3 * dot(grad4[gi3], x3, y3, z3, w3);
    }

    float t4 = 0.6f - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4;
    if (t4 < 0) {
        n4 = 0.0f;
    }
    else {
        t4 *= t4;
        n4 = t4 * t4 * dot(grad4[gi4], x4, y4, z4, w4);
    }

    return 27.0f * (n0 + n1 + n2 + n3 + n4);
}
