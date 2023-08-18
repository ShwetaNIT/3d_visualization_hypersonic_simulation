#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <set>
#include <map>
#include <sstream>
#include <cmath>
#include <assert.h>
#include <float.h>

// #define SHWETA

#include <GLFW/glfw3.h>
#include <GLUT/glut.h>
#include <OpenGL/glu.h>
#include <OpenGL/gl.h>
#include "CycleTable.h"

#ifdef SHWETA
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <GL/glew.h>
#else
//#include <Windows.h>
//#include "glut.h"
//#include <gl/gl.h>
//#include <gl/glu.h>
//#include <omp.h>
//#include "Timer.h"
#endif
#include "CycleTable.h"

using namespace std;

int WINDOW_WIDTH = 500, WINDOW_HEIGHT = 500;

long double MIN_PRESSURE_VALUE=0;

bool drawOverlayFlag = false;
bool drawCurvature = true;


// mouse state
int prevX = 0, prevY = 0;
bool leftPressed = false, rightPressed = false, middlePressed = false;

// view state
float rotMat [ 16 ] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
float focus [ 3 ] = {0,0,-1.5};

int faces [ 6 ] [ 4 ] = {{0,2,6,4},{1,3,7,5},{0,4,5,1},{2,6,7,3},{0,1,3,2},{4,5,7,6}};
int edges [ 12 ] [ 2 ] = {{0,1},{2,3},{6,7},{4,5},{0,2},{4,6},{5,7},{1,3},{0,4},{1,5},{3,7},{2,6}};


vector<long double> values;

class GridPoint {
    private:
        long double x;
        long double y;
        long double z;
        long double value;
        bool visited;
    public:
        GridPoint(long double x1, long double y1, long double z1, long double value1){
            x=x1;
            y=y1;
            z=z1;
            value=value1;
            visited = false;
        }
        GridPoint(){
            x=0;
            y=0;
            z=0;
            value=0;
            visited = false;
        }
        GridPoint(long double x1, long double y1, long double z1){
            x=x1;
            y=y1;
            z=z1;
            value=0.0;
            visited = false;
        }
        void setValues(long double x1, long double y1, long double z1, long double value1){
            x=x1;
            y=y1;
            z=z1;
            value=value1;
        }
        void setvisited(bool vis){
            visited= vis;
        }
        bool getvisited() const{
            return visited;
        }
        long double getx() const{
            return x;
        }
        long double gety() const{
            return y;
        }
        long double getz() const{
            return z;
        }
        void setx(long double x1){
            x=x1;
        }
        void sety(long double y1){
            y=y1;
        }
        void setz(long double z1){
            z=z1;
        }
        long double getvalue() const{
            return value;
        }
        GridPoint operator - (const GridPoint& theOther) const {
            GridPoint rvalue;
            rvalue.x = x - theOther.x;
            rvalue.y = y - theOther.y;
            rvalue.z = z - theOther.z;

            return rvalue;
        }

        GridPoint operator + (const GridPoint& theOther) const {
            GridPoint rvalue;
            rvalue.x = x + theOther.x;
            rvalue.y = y + theOther.y;
            rvalue.z = z + theOther.z;

            return rvalue;
        }

        GridPoint operator / (const long double& s) const {
            GridPoint rvalue;
            rvalue.x = x / s;
            rvalue.y = y / s;
            rvalue.z = z / s;

            return rvalue;
        }

        GridPoint operator * (const long double& s) const {
            GridPoint rvalue;
            rvalue.x = x * s;
            rvalue.y = y * s;
            rvalue.z = z * s;

            return rvalue;
        }

        GridPoint cross(const GridPoint& theOther) const {
            GridPoint rvalue;

            rvalue.x = theOther.z * y - theOther.y * z;
            rvalue.y = -theOther.z * x + theOther.x * z;
            rvalue.z = theOther.y * x - theOther.x * y;

            return rvalue;
        }

        long double dot(const GridPoint& theOther) const {
            return (x*theOther.x + y*theOther.y + z*theOther.z);
        }

        long double len() const {
            return sqrt(x*x + y*y + z*z);
        }

        void normalize(void) {
            long double len = sqrt(x * x + y * y + z * z);
            x /= len;
            y /= len;
            z /= len;
        }
        bool operator == (const GridPoint& theOther) const {
        // compare titles
        return x == theOther.x && y == theOther.y && z == theOther.z;
        }
};

bool operator < (const GridPoint& first, const GridPoint& second)
{
    return first.getvalue() <second.getvalue();
}

vector<GridPoint> polygonVertices;
vector<vector<int>> polygons;

vector<vector<vector<GridPoint>>> grid1;
vector<vector<vector<GridPoint>>> grid2;
map<GridPoint, vector<int>> gridMap;
vector<GridPoint> ridgePoints;

vector<long double> curvature;
vector<GridPoint> curvatureNormal;
vector<GridPoint> curvatureNormalBoundary;
vector<int> val;
void hueToRGB(long double hue, float& r, float& g, float& b)
{
    float colors[5][3] = { {1, 0, 0}, { 1,.5,0 }, { 1,1,0 }, { 0,1,0 }, { 0,0,1 } };
    if (hue < 0.25)
    {
        r = (colors[0][0] * (.25 - hue) + colors[1][0] * (hue - 0)) / .25;
        g = (colors[0][1] * (.25 - hue) + colors[1][1] * (hue - 0)) / .25;
        b = (colors[0][2] * (.25 - hue) + colors[1][2] * (hue - 0)) / .25;
    }
    else if (hue < 0.5)
    {
        r = (colors[1][0] * (.5 - hue) + colors[2][0] * (hue - 0.25)) / .25;
        g = (colors[1][1] * (.5 - hue) + colors[2][1] * (hue - 0.25)) / .25;
        b = (colors[1][2] * (.5 - hue) + colors[2][2] * (hue - 0.25)) / .25;
    }
    else if (hue < 0.75)
    {
        r = (colors[2][0] * (.75 - hue) + colors[3][0] * (hue - 0.5)) / .25;
        g = (colors[2][1] * (.75 - hue) + colors[3][1] * (hue - 0.5)) / .25;
        b = (colors[2][2] * (.75 - hue) + colors[3][2] * (hue - 0.5)) / .25;
    }
    else
    {
        r = (colors[3][0] * (1 - hue) + colors[4][0] * (hue - 0.75)) / .25;
        g = (colors[3][1] * (1 - hue) + colors[4][1] * (hue - 0.75)) / .25;
        b = (colors[3][2] * (1 - hue) + colors[4][2] * (hue - 0.75)) / .25;
    }
}

void calcValence(void)
{
    val.clear();
    val.resize(polygonVertices.size(), 0);

    for (int i = 0; i < polygons.size(); i++)
    {
        for (int j = 0; j < polygons[i].size(); j++) {
            val[polygons[i][j]]++;
        }
    }
}

void calculateMeanCurvatureNormal(void){
    curvatureNormal.clear();
    GridPoint x;
    curvatureNormal.resize(polygonVertices.size(), x);


    for (int i = 0; i < polygons.size(); i++)
    {
        GridPoint a = polygonVertices[polygons[i][0]];
        for (int j = 1; j < polygons[i].size() - 1; j++)
        {
            GridPoint b = polygonVertices[polygons[i][j]];
            GridPoint c = polygonVertices[polygons[i][j + 1]];
            GridPoint n = (a - b).cross(c - b);
            n.normalize();
            GridPoint pb = (c - a).cross(n);
            GridPoint pa = (b - c).cross(n);
            GridPoint pc = (a - b).cross(n);
            curvatureNormal[polygons[i][j]] = curvatureNormal[polygons[i][j]] + pb;
            curvatureNormal[polygons[i][0]] = curvatureNormal[polygons[i][0]] + pa;
            curvatureNormal[polygons[i][j + 1]] = curvatureNormal[polygons[i][j + 1]] + pc;
        }
    }
}

typedef enum
{
    INTERIOR,
    BOUNDARY,
    CORNER
} VertexType;

vector<VertexType> vertType;
vector<vector<int>> boundaryVerticesNeighbors;

void boundaryLaplacianSmoothing(void){
        for(int i=0;i<boundaryVerticesNeighbors.size();i++){
        if(boundaryVerticesNeighbors[i].size() == 2 && vertType[i]==BOUNDARY){
            GridPoint vec1 = (polygonVertices[boundaryVerticesNeighbors[i][0]]-polygonVertices[i]);
            //vec1 = vec1/vec1.len();
            GridPoint vec2 = (polygonVertices[boundaryVerticesNeighbors[i][1]]-polygonVertices[i]);
            //vec2 = vec2/vec2.len();
            curvatureNormalBoundary[i] = vec1+vec2;
        }else if(vertType[i]==BOUNDARY){
            //This code shouldn't be executed
            cout<<i<<endl;
        }
    }
}

void laplacianSmoothingBoundaryVertices(void) {
    curvatureNormalBoundary.clear();
    GridPoint x;
    curvatureNormalBoundary.resize(polygonVertices.size(), x);
    vertType.clear();
    vertType.resize(polygonVertices.size(), INTERIOR);
    boundaryVerticesNeighbors.clear();
    vector<int> neigh;
    boundaryVerticesNeighbors.resize(polygonVertices.size(), neigh);
    for (int i = 0; i < polygonVertices.size(); i++) {
        if (val[i] == 1) {
            vertType[i] = CORNER;
        }
        else if (val[i] == 2)
        {
            vertType[i] = BOUNDARY;
        }
        else{
            vertType[i] = INTERIOR;
        }
    }

    for (int i = 0; i < polygons.size(); i++) {
        for (int j = 0; j < polygons[i].size(); j++) {
            int first = polygons[i][j];
            int second = polygons[i][(j + 1) % polygons[i].size()];

            if (vertType[first] == BOUNDARY && vertType[second] != INTERIOR)
            {
                vector<int>::iterator it = find(boundaryVerticesNeighbors[first].begin(),boundaryVerticesNeighbors[first].end(), second);
                if (it == boundaryVerticesNeighbors[first].end())
                {
                    boundaryVerticesNeighbors[first].push_back(second);
                }
                else
                {
                    boundaryVerticesNeighbors[first].erase(it);
                }
            }

            if (vertType[second] == BOUNDARY && vertType[first] != INTERIOR)
            {
                vector<int>::iterator it = find(boundaryVerticesNeighbors[second].begin(), boundaryVerticesNeighbors[second].end(), first);
                if (it == boundaryVerticesNeighbors[second].end())
                {
                    boundaryVerticesNeighbors[second].push_back(first);
                }
                else
                {
                    boundaryVerticesNeighbors[second].erase(it);
                }
            }
        }
    }
    boundaryLaplacianSmoothing();

}



void computeMeanCurvature(void)
{
    curvature.clear();
    curvature.resize(polygonVertices.size(), 0);
    val.resize(polygonVertices.size(), 0);
    calcValence();
    calculateMeanCurvatureNormal();
    laplacianSmoothingBoundaryVertices();

    for(int i=0;i<polygonVertices.size();i++){
        if(val[i]<=2){
        curvature[i]=0.0;
        }
        else
        curvature[i] = curvatureNormal[i].len();
    }


    int i = 0;
    long double minC, maxC;

    for (i = 0; val[i] <= 2 && i < polygonVertices.size(); i++){

    }

    minC = maxC = curvature[i];
    i++;
    for (; i < polygonVertices.size(); i++)
    {
        if (val[i] <= 2)
        {
            continue;
        }
        if (curvature[i] > maxC)
        {
            maxC = curvature[i];
        }
        if (curvature[i] < minC)
        {
            minC = curvature[i];
        }
    }
    cout<<"Min C: "<<minC<<" Max C: "<<maxC<<endl;
    cout << "Min curvature: " << (2 * 3.141592654 - maxC) * 360 / (2 * 3.141592654) << endl << "Max curvature: " << (2 * 3.141592654 - minC) * 360 / (2 * 3.141592654) << endl;
    long double oldRange = maxC-minC;
    long double newRange = 9.0;

    for (int i = 0; i < curvature.size(); i++)
    {

        /*
        OldRange = (OldMax - OldMin)  
        NewRange = (NewMax - NewMin)  
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
        */
        if(val[i]>=2)
        curvature[i] = log10((((curvature[i]-minC)*newRange)/oldRange)+1.0);
        else
        curvature[i] = 0.0;
    }

}



void laplacianSmoothing(void) {
    curvatureNormal.clear();
    GridPoint x;
    curvatureNormal.resize(polygonVertices.size(), x);
    val.resize(polygonVertices.size(), 0);
    calcValence();
    for (int k = 0; k < 1000; k++) {
        calculateMeanCurvatureNormal();
        for (int i = 0; i < polygonVertices.size(); i++) {

            //if(curvatureNormal[i].len()!=0)
            //curvatureNormal[i] = curvatureNormal[i]/curvatureNormal[i].len();
            //curvatureNormal[i] = curvatureNormal[i]*0.01;
            if (val[i] > 2)
            {
                polygonVertices[i] = polygonVertices[i] + (curvatureNormal[i] * 0.01);
            }
        }
    }
    //Laplacian Smoothing of Boundary Vertices
    for (int k = 0; k < 1000; k++) {
        boundaryLaplacianSmoothing();
        for (int i = 0; i < polygonVertices.size(); i++) {
            if (vertType[i]==BOUNDARY)
            {
                polygonVertices[i] = polygonVertices[i] + (curvatureNormalBoundary[i] * 0.001);
            }
        }
    }

    computeMeanCurvature();
}

void computeGaussianCurvature(void)
{
    curvature.clear();
    curvature.resize(polygonVertices.size(), 0);
    int* val = new int[polygonVertices.size()];
    memset(val, 0, sizeof(int) * polygonVertices.size());
    for (int i = 0; i < polygons.size(); i++)
    {
        for (int j = 0; j < polygons[i].size(); j++)
        {
            GridPoint a = polygonVertices[polygons[i][(j - 1 + polygons[i].size()) % polygons[i].size()]];
            GridPoint b = polygonVertices[polygons[i][j]];
            GridPoint c = polygonVertices[polygons[i][(j + 1) % polygons[i].size()]];
            curvature[polygons[i][j]] += acos((a - b).dot(c - b) / ((a - b).len() * (c - b).len()));
            if (!isfinite(curvature[polygons[i][j]]))
            {
                cout << "problem here" << endl;
            }
            val[polygons[i][j]]++;
        }
    }
    int i = 0;
    long double minC, maxC;

    for (i = 0; val[i] != 4 && i < polygonVertices.size(); i++){

    }

    minC = maxC = curvature[i];
    i++;
    for (; i < polygonVertices.size(); i++)
    {
        if (val[i] != 4)
        {
            continue;
        }
        if (curvature[i] > maxC)
        {
            maxC = curvature[i];
        }
        if (curvature[i] < minC)
        {
            minC = curvature[i];
        }
    }
    cout << "Min curvature: " << (2 * 3.141592654 - maxC) * 360 / (2 * 3.141592654) << endl << "Max curvature: " << (2 * 3.141592654 - minC) * 360 / (2 * 3.141592654) << endl;
    for (int i = 0; i < curvature.size(); i++)
    {
        if (val[i] != 4)
        {
            curvature[i] = 2 * 3.141592654; // set to flat if a boundary vertex
        }
        curvature[i] = (curvature[i] - minC) / (maxC - minC);
    }
    delete[] val;
}

void constructCube(vector<GridPoint>& vec, int &i, int &j,int &k, vector<vector<vector<GridPoint>>>& grid){

    vec.push_back(grid[i][j][k]); //0
    vec.push_back(grid[i][j+1][k]);//1
    vec.push_back(grid[i+1][j][k]);//2
    vec.push_back(grid[i+1][j+1][k]);//3
    vec.push_back(grid[i][j][k+1]); //4
    vec.push_back(grid[i][j+1][k+1]);//5
    vec.push_back(grid[i+1][j][k+1]);//6
    vec.push_back(grid[i+1][j+1][k+1]);//7

}

int calculateCubeIndex(vector<GridPoint>& vec, long double &MIN_PRESSURE_VALUE){
    int cubeIndex = 0;
    for (int i = 0; i < 8; i++)
        if (vec[i].getvalue() > MIN_PRESSURE_VALUE) cubeIndex |= (1 << i);
    return cubeIndex;
}
/*
int mapGridPoint(GridPoint& gridPoint, int& maxDim, map<GridPoint, vector<int>>& gridMap){
    auto it = gridMap.find(gridPoint);
    return it->second[0]*maxDim*maxDim + it->second[1]*maxDim + it->second[2];
}
*/



void constructPolygons(vector<vector<vector<GridPoint>>>& grid, long double& MIN_PRESSURE_VALUE){

    //Part 2 of the problem

    int m = grid.size();
    int n = grid[0].size();
    int p = grid[0][0].size();

    int maxDim = m+n+p;

    int badPolygons = 0;

    if (&grid == &grid1)
    {
        ridgePoints.clear();
    }
    for (int i = 1; i < m - 2; i++) {
        for (int j = 1; j < n - 2; j++) {
            for (int k = 1; k < p - 2; k++)
            {
                long double a, b, c, d, qa, qb, qc;
                a = grid[i - 1][j][k].getvalue();
                b = grid[i][j][k].getvalue();
                c = grid[i + 1][j][k].getvalue();
                d = grid[i + 2][j][k].getvalue();

                qa = 6 * c - 2 * a - 3 * b - d;
                qb = a - 9 * b + 9 * c - d;
                qc = a - 6 * b + 3 * c + 2 * d;

                if ((qb - qa <= 0 && qc - qb > 0) || (qb - qa > 0 && qc - qb <= 0))
                {
                    if ( fabs (qa - 2 * qb + qc) > 0.001 && (qa * qc - qb * qb) / (qa - 2 * qb + qc) < 0)
                    {
                        ridgePoints.push_back(grid[i][j][k] * ((qc - qb) / (qa - 2 * qb + qc)) + grid[i + 1][j][k] * ((qa - qb) / (qa - 2 * qb + qc)));
                    }
                }
                a = grid[i][j-1][k].getvalue();
                b = grid[i][j][k].getvalue();
                c = grid[i][j+1][k].getvalue();
                d = grid[i][j+2][k].getvalue();

                qa = 6 * c - 2 * a - 3 * b - d;
                qb = a - 9 * b + 9 * c - d;
                qc = a - 6 * b + 3 * c + 2 * d;

                if ((qb - qa <= 0 && qc - qb > 0) || (qb - qa > 0 && qc - qb <= 0))
                {
                    if (fabs(qa - 2 * qb + qc) > 0.001 && (qa * qc - qb * qb) / (qa - 2 * qb + qc) < 0)
                    {
                        ridgePoints.push_back(grid[i][j][k] * ((qc - qb) / (qa - 2 * qb + qc)) + grid[i][j+1][k] * ((qa - qb) / (qa - 2 * qb + qc)));
                    }
                }

                a = grid[i][j][k-1].getvalue();
                b = grid[i][j][k].getvalue();
                c = grid[i][j][k+1].getvalue();
                d = grid[i][j][k+2].getvalue();

                qa = 6 * c - 2 * a - 3 * b - d;
                qb = a - 9 * b + 9 * c - d;
                qc = a - 6 * b + 3 * c + 2 * d;

                if ((qb - qa <= 0 && qc - qb > 0) || (qb - qa > 0 && qc - qb <= 0))
                {
                    if (fabs(qa - 2 * qb + qc) > 0.001 && (qa * qc - qb * qb) / (qa - 2 * qb + qc) < 0)
                    {
                        ridgePoints.push_back(grid[i][j][k] * ((qc - qb) / (qa - 2 * qb + qc)) + grid[i][j][k+1] * ((qa - qb) / (qa - 2 * qb + qc)));
                    }
                }
            }
        }
    }



    map<pair<int,int>, int> edgeToVertex;



    for(int i=0;i<m-1;i++){
        for(int j=0;j<n-1;j++){
            for(int k=0;k<p-1;k++){
                vector<GridPoint> cube;
                constructCube(cube, i,j,k, grid);
                int cubeIndex = calculateCubeIndex(cube, MIN_PRESSURE_VALUE);
                vector< vector<int> > &cycles = CycleTable::singleton ( )->getCycles ( cubeIndex );

                for (int q = 0; q < cycles.size(); q++)
                {
                    vector<int> polygon;
                    for (int r = 0; r < cycles[q].size(); r++)
                    {
                        int start = edges[cycles[q][r]][0];
                        int end = edges[cycles[q][r]][1];
                        GridPoint startPoint = cube[start];
                        int startVal = (i + ((start >> 1) & 0x01)) * maxDim * maxDim + (j + (start & 0x01)) * maxDim + k + ((start >> 2) & 0x01);
                        GridPoint endPoint = cube[end];
                        int endVal = (i + ((end >> 1) & 0x01)) * maxDim * maxDim + (j + (end & 0x01)) * maxDim + k + ((end >> 2) & 0x01);
                        int lessVal = 0;
                        int moreVal = 0;

                        if (startVal < endVal) {
                            lessVal = startVal;
                            moreVal = endVal;
                        }
                        else {
                            lessVal = endVal;
                            moreVal = startVal;
                        }

                        long double epsilon = 0;// 0.00000000001;
                        long double v1 = startPoint.getvalue();
                        long double v2 = endPoint.getvalue();
                        if (fabs(MIN_PRESSURE_VALUE - v1) < epsilon)
                        {
                            v1 = MIN_PRESSURE_VALUE;
                        }
                        if (fabs(MIN_PRESSURE_VALUE - v2) < epsilon)
                        {
                            v2 = MIN_PRESSURE_VALUE;
                        }
                        long double alpha = (MIN_PRESSURE_VALUE - v1) / (v2 - v1);

                        pair<int, int> key = make_pair(lessVal, moreVal);

                        if (v1 == MIN_PRESSURE_VALUE)
                        {
                            key = make_pair(startVal, -1);
                        }
                        else if (v2 == MIN_PRESSURE_VALUE)
                        {
                            key = make_pair(endVal, -1);
                        }

                        auto b = edgeToVertex.find(key);
                        int ind;
                        if (b != edgeToVertex.end()) {
                           ind = b->second;
                        }
                        else
                        {
                            long double x = (endPoint.getx() * alpha + startPoint.getx() * (1 - alpha));
                            long double y = (endPoint.gety() * alpha + startPoint.gety() * (1 - alpha));
                            long double z = (endPoint.getz() * alpha + startPoint.getz() * (1 - alpha));
                            polygonVertices.push_back(GridPoint(x, y, z));
                            ind = polygonVertices.size() - 1;
                            edgeToVertex[key] = polygonVertices.size() - 1;
                        }
                        polygon.push_back(ind);
                    }
                    vector<int> newPoly;
                    for (int r = 0; r < polygon.size(); r++)
                    {
                        if (polygon[r] != polygon[(r + 1) % polygon.size()])
                        {
                            newPoly.push_back(polygon[r]);
                        }
                        else
                        {
                            badPolygons++;
                        }
                    }
                    if (newPoly.size() > 2)
                    {
                        polygons.push_back(newPoly);
                    }
                }
            }
        }
    }
    
    cout<<"Total polygons we are displaying: "<<polygons.size()<<endl;
    cout<<"Bad polygons for the grid are: "<<badPolygons<<endl;
}

void drawMesh(){
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    float r, g, b;
    glColor3f(1, 1, 1);
   for(int i=0;i<polygons.size();i++){
    glBegin(GL_TRIANGLES);
    GridPoint first = polygonVertices[polygons[i][0]];
    for(int j=1;j<polygons[i].size()-1;j++){
        GridPoint second = polygonVertices[polygons[i][j]];
        GridPoint third = polygonVertices[polygons[i][j+1]];
        GridPoint normal = (second - first).cross(third - first);
        normal.normalize();
        glNormal3d(normal.getx(), normal.gety(), normal.getz());
        if (drawCurvature)
        {
            hueToRGB(curvature[polygons[i][0]], r, g, b);
            glColor3f(r, g, b);
        }
        glVertex3d(first.getx(), first.gety(), first.getz());
        if (drawCurvature)
        {
            hueToRGB(curvature[polygons[i][j]], r, g, b);
            glColor3f(r, g, b);
        }
        glVertex3d(second.getx(), second.gety(), second.getz());
        if (drawCurvature)
        {
            hueToRGB(curvature[polygons[i][j + 1]], r, g, b);
            glColor3f(r, g, b);
        }
        glVertex3d(third.getx(), third.gety(), third.getz());
    }

    glEnd();

   }
}

void drawOverlay(){
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE);
   for(int i=0;i<polygons.size();i++){
    glBegin(GL_POLYGON);
    for(int j=0;j<polygons[i].size();j++){
        GridPoint p = polygonVertices[polygons[i][j]];
        glVertex3d(p.getx(), p.gety(), p.getz());
    }
    glEnd();
   }
}

void display() 
{


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set up viewing matrices
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, float(WINDOW_WIDTH)/WINDOW_HEIGHT, .0001, 100);
	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE); 

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);

	//Camera
	glTranslatef(focus[0], focus[1], focus[2]);
	glMultMatrixf(rotMat);

    glDisable ( GL_CULL_FACE ); 
    //glPolygonMode( GL_FRONT_AND_BACK, GL_LINE);

    //Drawing polygons
    drawMesh();
    /*
     glColor3f(1, 1, 1);
    glBegin(GL_LINES);


    for (int i = 0; i < polygonVertices.size(); i++)
    {
        if (val[i] <=2)
        {
            glVertex3f(polygonVertices[i].getx(), polygonVertices[i].gety(), polygonVertices[i].getz());
            glVertex3f(polygonVertices[i].getx(), polygonVertices[i].gety()+0.001, polygonVertices[i].getz());
        }
    }


    for(auto it = boundaryVerticesNeighbors.begin();it!=boundaryVerticesNeighbors.end();it++){
        if(it->second.size()>2){
            for(auto ip = it->second.begin();ip!=it->second.end();ip++){
                glVertex3f(polygonVertices[it->first].getx(), polygonVertices[it->first].gety(), polygonVertices[it->first].getz());
                glVertex3f(polygonVertices[*ip].getx(), polygonVertices[*ip].gety(), polygonVertices[*ip].getz());
                //glVertex3f(polygonVertices[*ip].getx(), polygonVertices[*ip].gety()+0.01, polygonVertices[*ip].getz());
            }
        }

    }
    
    

    glEnd();
    */
    
    glColor3f(0, 0, 0);
    glBegin(GL_POINTS);
    for (int i = 0; i < ridgePoints.size(); i++)
    {
        glVertex3d(ridgePoints[i].getx(), ridgePoints[i].gety(), ridgePoints[i].getz());
    }
    glEnd();
    
      
      

    if(drawOverlayFlag){
        glColor3f ( 0, 0, 0 );
        //glDisable ( GL_LIGHTING );
        glPolygonOffset ( -1, -1 );
        glEnable ( GL_POLYGON_OFFSET_LINE );
        glEnable ( GL_POLYGON_OFFSET_FILL );
        drawOverlay(); // note that you shouldn’t use glColor3f in this draw and you don’t need normal either
        glDisable ( GL_POLYGON_OFFSET_LINE );
        glDisable ( GL_POLYGON_OFFSET_FILL );
        //glEnable ( GL_LIGHTING );
    }


	glFlush ();
	glutSwapBuffers();
}

void reshape(int width, int height)
{
	glViewport(0, 0, width, height);
	WINDOW_WIDTH = width;
	WINDOW_HEIGHT = height;
}

bool uppressed(int x, int y){
    return ((x>1200) && (y>700));
}

bool downpressed(int x, int y){
    return ((x>1200) && (y<100));
}

void mouse(int button, int state, int x, int y)
{
	y = WINDOW_HEIGHT - y;
    /*

    if(state == GLUT_DOWN && button == GLUT_LEFT_BUTTON && uppressed(x,y)){
        MIN_PRESSURE_VALUE*=pow(1.01,10);
        polygonVertices.clear();
        polygons.clear();
        constructPolygons(grid1, gridMap, MIN_PRESSURE_VALUE);
        constructPolygons(grid2, gridMap, MIN_PRESSURE_VALUE);
        computeMeanCurvature();
        glutPostRedisplay();
    }

    if(state == GLUT_DOWN && button == GLUT_LEFT_BUTTON && downpressed(x,y)){
        MIN_PRESSURE_VALUE/=pow(1.01,10);
        polygonVertices.clear();
        polygons.clear();
        constructPolygons(grid1, gridMap, MIN_PRESSURE_VALUE);
        constructPolygons(grid2, gridMap, MIN_PRESSURE_VALUE);
        computeMeanCurvature();
        glutPostRedisplay();
    }
    */

	// Mouse state that should always be stored on pressing
	if (state == GLUT_DOWN)
	{
		prevX = x;
		prevY = y;
	}

	if (button == GLUT_LEFT_BUTTON)
	{
		leftPressed = state == GLUT_DOWN;
	}

	if (button == GLUT_RIGHT_BUTTON)
	{
		rightPressed = state == GLUT_DOWN;
	}
	
	if (button == GLUT_MIDDLE_BUTTON)
	{
		middlePressed = state == GLUT_DOWN;
	}
}

void motion(int x, int y)
{
	y = WINDOW_HEIGHT - y;

	float dx = (x - prevX);
	float dy = (y - prevY);

	// rotate the scene
	if (leftPressed)
	{
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glRotatef(dx, 0, 1, 0);
		glRotatef(dy, -1, 0, 0);
		glMultMatrixf(rotMat);
		glGetFloatv(GL_MODELVIEW_MATRIX, rotMat);
	}
	else if (middlePressed)
	{
		focus [ 0 ] += 0.005 * dx;
		focus [ 1 ] += 0.005 * dy;
	}
	else if (rightPressed)
	{
		focus [ 2 ] += 0.01 * dy;
	}
	
	// Store previous mouse positions
	prevX = x;
	prevY = y;

	glutPostRedisplay();
}


void keyboard(unsigned char key, int x, int y)
{
    FILE* fptr;

    switch(key){
        case ' ':
            drawOverlayFlag = !drawOverlayFlag;
            glutPostRedisplay();
            break;
        case 'l':
        case 'L':
            laplacianSmoothing();
            glutPostRedisplay();
            break;
        case 'c':
        case 'C':
            drawCurvature = !drawCurvature;
            glutPostRedisplay();
            break;
        case '8':
            fptr = fopen("rotmat.bin", "wb");
            fwrite(rotMat, sizeof(float), 16, fptr);
            fwrite(focus, sizeof(float), 3, fptr);
            fclose(fptr);
            break;
        case '9':
            fptr = fopen("rotmat.bin", "rb");
            fread(rotMat, sizeof(float), 16, fptr);
            fread(focus, sizeof(float), 3, fptr);
            fclose(fptr);
            glutPostRedisplay();
            break;
    }
}

void SpecialInput(int key, int x, int y)
{
    switch(key)
    {
    case GLUT_KEY_UP:
        MIN_PRESSURE_VALUE*=pow(1.01,10);
        cout << "Min pressure now " << MIN_PRESSURE_VALUE << endl;
        polygonVertices.clear();
        polygons.clear();
        constructPolygons(grid1, MIN_PRESSURE_VALUE);
        constructPolygons(grid2, MIN_PRESSURE_VALUE);
        computeMeanCurvature();
    break;
    case GLUT_KEY_DOWN:
        MIN_PRESSURE_VALUE/=pow(1.01,10);
        cout << "Min pressure now " << MIN_PRESSURE_VALUE << endl;
        polygonVertices.clear();
        polygons.clear();
        constructPolygons(grid1, MIN_PRESSURE_VALUE);
        constructPolygons(grid2, MIN_PRESSURE_VALUE);
        computeMeanCurvature();
    break;
    }

    glutPostRedisplay();
}

void init(void)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	glClearColor(0, 0, 0, 1);   
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	glEnable(GL_CULL_FACE);
}




long double stringToDouble(const std::string& str) {
    long double result = 0;
    long double multiplier = 1;
    int exponent = 0;
    bool isNegative = false;
    bool afterdecimal=false;

    int i = 0;
    if (str[0] == '-') {
        isNegative = true;
        i++;
    }
    for (; i < str.length(); i++) {
        char c = str[i];
        if (c == 'e' || c == 'E') {
            exponent = std::stoi(str.substr(i + 1));
            break;
        } else if (c == '.') {
            afterdecimal=true;
        } else {
            result = result * 10 + (c - '0');
            if(afterdecimal == true){
                multiplier *= 10;
            }
        }
    }

    result /= multiplier;

    if (isNegative) {
        result = -result;
    }

    return result * std::pow(10, exponent);
}

void increaseDimensions(vector<vector<vector<GridPoint>>>& grid, int i, int j, int k){
    while(grid.size()<=i){
        grid.push_back(vector<vector<GridPoint>> (j+1, vector<GridPoint>(k+1)));
    }
    while(grid[i].size()<=j){
        grid[i].push_back(vector<GridPoint> (k+1));
    }
    while(grid[i][j].size()<=k){
        GridPoint gridpoint;
        grid[i][j].push_back(gridpoint);
    }
}

vector<string> split(string s, char delimiter) {
    vector<string> tokens;
    size_t start = 0, end = 0;
    while ((end = s.find(delimiter, start)) != string::npos) {
        tokens.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    tokens.push_back(s.substr(start));
    return tokens;
}

string converttostring(int i, int j, int k){
    string s="";
    s.append(to_string(i)+" ");
    s.append(to_string(j)+" ");
    s.append(to_string(k));
    return s;
}



void readFileIntoGrid(string& fname, vector<vector<vector<GridPoint>>>& grid){
    vector<string> row;
	string line, word;
 
	fstream file (fname, ios::in);

    int header = 1 ;// set zero if no header (header is the first row with column descriptions)

    //Change the value of rows if no header present i.e. header == 0 
    int rows = -1; // since we start populating the grid from row index 1 if header == 1. Therefore, row no 100 in excel maps to index 98 in grid

	if(file.is_open())
	{
		while(getline(file, line))
		{
            if(header==0){
                row.clear();
    
                stringstream str(line);

                //Getting the first four values since we are only concerned with x,y,z and density
                while(getline(str, word, ',')){
                    row.push_back(word);
                }

                //storing the point in grid object
                //0->value
                //1->x
                //2->y
                //3->z
                //4->k
                //5->j
                //6->i

                // increase dimensions of grid if necessary
                int i = stoi(row[6]);
                int j = stoi(row[5]);
                int k = stoi(row[4]);
                long double pressure = stringToDouble(row[0]);
                increaseDimensions(grid, i,j,k);
                grid[i][j][k].setValues(stringToDouble(row[1]), stringToDouble(row[2]), stringToDouble(row[3]), pressure);
                values.push_back(pressure);
//                gridMap[grid[i][j][k]] = vector<int>({i,j,k});
                
            } else {
                // skipping first row if header=1
                header=0;
            }
            rows++;
		}
	}
	else
		cout<<"Could not open the file\n";
}

long double avg = 0;
int num = 0;
void calcContourVal(vector<vector<vector<GridPoint>>>& grid)
{
    for (int i = 1; i < grid.size() - 2; i++) {
        for (int j = 1; j < grid[i].size() - 2; j++) {
            for (int k = 1; k < grid[i][j].size() - 2; k++)
            {
                long double a, b, c, d, qa, qb, qc;
                a = grid[i - 1][j][k].getvalue();
                b = grid[i][j][k].getvalue();
                c = grid[i + 1][j][k].getvalue();
                d = grid[i + 2][j][k].getvalue();

                qa = 6 * c - 2 * a - 3 * b - d;
                qb = a - 9 * b + 9 * c - d;
                qc = a - 6 * b + 3 * c + 2 * d;

                if ((qb - qa <= 0 && qc - qb > 0) || (qb - qa > 0 && qc - qb <= 0))
                {
                    if (fabs(qa - 2 * qb + qc) > 0.001 && (qa * qc - qb * qb) / (qa - 2 * qb + qc) < 0)
                    {
                        long double t = (qa - qb) / (qa - 2 * qb + qc);
                        avg += b * (1 - t) + c * t;
                        num++;
                    }
                }
                a = grid[i][j - 1][k].getvalue();
                b = grid[i][j][k].getvalue();
                c = grid[i][j + 1][k].getvalue();
                d = grid[i][j + 2][k].getvalue();

                qa = 6 * c - 2 * a - 3 * b - d;
                qb = a - 9 * b + 9 * c - d;
                qc = a - 6 * b + 3 * c + 2 * d;

                if ((qb - qa <= 0 && qc - qb > 0) || (qb - qa > 0 && qc - qb <= 0))
                {
                    if (fabs(qa - 2 * qb + qc) > 0.001 && (qa * qc - qb * qb) / (qa - 2 * qb + qc) < 0)
                    {
                        long double t = (qa - qb) / (qa - 2 * qb + qc);
                        avg += b * (1 - t) + c * t;
                        num++;
                    }
                }

                a = grid[i][j][k - 1].getvalue();
                b = grid[i][j][k].getvalue();
                c = grid[i][j][k + 1].getvalue();
                d = grid[i][j][k + 2].getvalue();

                qa = 6 * c - 2 * a - 3 * b - d;
                qb = a - 9 * b + 9 * c - d;
                qc = a - 6 * b + 3 * c + 2 * d;

                if ((qb - qa <= 0 && qc - qb > 0) || (qb - qa > 0 && qc - qb <= 0))
                {
                    if (fabs(qa - 2 * qb + qc) > 0.001 && (qa * qc - qb * qb) / (qa - 2 * qb + qc) < 0)
                    {
                        long double t = (qa - qb) / (qa - 2 * qb + qc);
                        avg += b * (1 - t) + c * t;
                        num++;
                    }
                }
            }
        }
    }
}
 
int main(void)
{

    vector<string> fnames({"block_1.csv","block_2.csv"});


    readFileIntoGrid(fnames[0], grid1);
//    cout<<"Total number of elements in gridmap: "<<gridMap.size()<<endl;
    readFileIntoGrid(fnames[1], grid2);

//    cout<<"Total number of elements in gridmap: "<<gridMap.size()<<endl;

    int m = grid1.size();
    int n = grid1[0].size();
    int p = grid1[0][0].size();

    long double totalSize = m*n*p;

    m = grid2.size();
    n = grid2[0].size();
    p = grid2[0][0].size();

    totalSize+=m*n*p;

    long double index = totalSize*0.01;

    sort(values.begin(), values.end());

    ////Looking at gridpoint at index
    //std::map<GridPoint, vector<int>>::iterator it = gridMap.begin();
    //std::advance(it, index);

    //int starti = (it->second)[0];
    //int startj = (it->second)[1];
    //int startk = (it->second)[2]; 


    //cout<<"Starting point: "<<starti<<" "<<startj<<" "<<startk<<endl;
    //cout<<"Original Pressure value : "<<(it->first).getvalue()<<endl;
    //MIN_PRESSURE_VALUE = (it->first).getvalue()*pow(1.01,55);

    MIN_PRESSURE_VALUE = values[values.size() * 0.7];
    calcContourVal(grid1);
    calcContourVal(grid2);
    MIN_PRESSURE_VALUE = avg / num;
    cout << "Pressure value we are looking at : " << MIN_PRESSURE_VALUE << endl;

    int gridno = 0;
    constructPolygons(grid1, MIN_PRESSURE_VALUE);
    gridno++;
    constructPolygons(grid2, MIN_PRESSURE_VALUE);
    //computeGaussianCurvature();
    computeMeanCurvature();

    

   //OpenGL code


    int argc = 1;
    char *argv[1] = {(char*)"Something"};
    glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(0, 0);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutCreateWindow("Shockwave Visualizer");

	init();

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
    glutSpecialFunc(SpecialInput);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	// main loop
	glutMainLoop();
    return 0;
}
 
 