//
//  main.cpp
//

#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <mpi.h>

using namespace std;
using namespace MPI;


struct mpi_double_int {
    double value;
    int location;
};

// Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
void invertSequential(Matrix& iA) {

    // vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    // construire la matrice [A I]
    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    // traiter chaque rangée
    for (size_t k = 0; k < iA.rows(); ++k) {
        // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
        // (pour une meilleure stabilité numérique).
        size_t p = k;
        double lMax = fabs(lAI(k, k));
        for (size_t i = k; i < lAI.rows(); ++i) {
            if (fabs(lAI(i, k)) > lMax) {
                lMax = fabs(lAI(i, k));
                p = i;
            }
        }
        // vérifier que la matrice n'est pas singulière
        if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

        // échanger la ligne courante avec celle du pivot
        if (p != k) lAI.swapRows(p, k);

        double lValue = lAI(k, k);
        for (size_t j = 0; j < lAI.cols(); ++j) {
            // On divise les éléments de la rangée k
            // par la valeur du pivot.
            // Ainsi, lAI(k,k) deviendra égal à 1.
            lAI(k, j) /= lValue;
        }

        // Pour chaque rangée...
        for (size_t i = 0; i < lAI.rows(); ++i) {
            if (i != k) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
                double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k) * lValue;
            }
        }

        // cout << "Matrice " << k << ": \n" << lAI.str() << endl;
    }

    // On copie la partie droite de la matrice AI ainsi transformée
    // dans la matrice courante (this).
    for (unsigned int i = 0; i < iA.rows(); ++i) {
        iA.getRowSlice(i) = lAI.getDataArray()[slice(i * lAI.cols() + iA.cols(), iA.cols(), 1)];
    }
}

void GatherResults(int lRank, MatrixConcatCols& lAI, int lSize, Matrix& pMatrix, Matrix& iA)
{
    if (lRank == 0) {
        double* recv = new double[(int)(ceil(lAI.rows() / (float)lSize)) * lAI.cols() * lSize];
        MPI_Gather((void*)&pMatrix(0, 0), (int)ceil(lAI.rows() / (float)lSize) * lAI.cols(), MPI_DOUBLE, recv, (int)ceil(lAI.rows() / (float)lSize) * lAI.cols(), MPI_DOUBLE, 0, COMM_WORLD);
        for (int i = 0; i < iA.rows(); ++i) {
            for (int j = 0; j < iA.cols(); ++j) {
                iA(i, j) = recv[j + i * lAI.cols() + iA.cols()];
            }
        }
    }
    else {
        MPI_Gather((void*)&pMatrix(0, 0), (int)ceil(lAI.rows() / (float)lSize) * lAI.cols(), MPI_DOUBLE, NULL, (int)ceil(lAI.rows() / (float)lSize) * lAI.cols(), MPI_DOUBLE, 0, COMM_WORLD);
    }
}

void FindPivot(Matrix& pMatrix, int lRank, const size_t& k, mpi_double_int& gMax)
{
    mpi_double_int lMax;
    lMax.value = 0;
    lMax.location = 0;

    for (size_t i = 0; i < pMatrix.rows(); i++) {
        if (lRank * pMatrix.rows() + i >= k && fabs(pMatrix(i, k)) > lMax.value) {
            lMax.value = fabs(pMatrix(i, k));
            lMax.location = (int)i + lRank * pMatrix.rows();
        }
    }

    MPI_Allreduce(&lMax, &gMax, 1, MPI_DOUBLE_INT, MPI_MAXLOC, COMM_WORLD);
}

void ApplyPivot(Matrix& pMatrix, int lRank, const size_t& k, double* rowPivot)
{
    for (int i = 0; i < pMatrix.rows(); ++i) {
        if (!(lRank * pMatrix.rows() <= k && k < (lRank + 1) * pMatrix.rows() && i == k % pMatrix.rows())) {
            double lValue = pMatrix(i, k);

            for (int j = 0; j < pMatrix.cols(); j++) {
                pMatrix(i, j) -= rowPivot[j] * lValue;
            }
        }
    }
}

void SplitMatrix(Matrix& pMatrix, int lRank, MatrixConcatCols& lAI)
{
    int u = 0;
    while (u < pMatrix.rows()) {
        if (u + lRank * pMatrix.rows() < lAI.rows()) {
            pMatrix.getRowSlice(u) = lAI.getRowCopy(u + lRank * pMatrix.rows());
        }
        u++;
    }
}

void SwapAndBroadcastPivot(int lRank, mpi_double_int& gMax, Matrix& pMatrix, const size_t& k, double* rowPivot)
{
    if (lRank == gMax.location / pMatrix.rows()) {
        double lValue = pMatrix(gMax.location % pMatrix.rows(), k);
        for (int j = 0; j < pMatrix.cols(); j++) {
            pMatrix(gMax.location % pMatrix.rows(), j) /= lValue;
            rowPivot[j] = pMatrix(gMax.location % pMatrix.rows(), j);
        }

        MPI_Bcast(rowPivot, pMatrix.cols(), MPI_DOUBLE, gMax.location / pMatrix.rows(), COMM_WORLD);
        if (lRank * pMatrix.rows() <= k && k < (lRank + 1) * pMatrix.rows()) {
            pMatrix.swapRows(k % pMatrix.rows(), gMax.location % pMatrix.rows());
        }
        else {
            Status lStatus;
            double swapRow[pMatrix.cols()];
            COMM_WORLD.Recv(&swapRow, pMatrix.cols(), MPI_DOUBLE, k / pMatrix.rows(), 1, lStatus);
            for (int j = 0; j < pMatrix.cols(); j++) {
                pMatrix(gMax.location % pMatrix.rows(), j) = swapRow[j];
            }
        }
    }
    else {
        MPI_Bcast(rowPivot, pMatrix.cols(), MPI_DOUBLE, gMax.location / pMatrix.rows(), COMM_WORLD);

        if (lRank * pMatrix.rows() <= k && k < (lRank + 1) * pMatrix.rows()) {
            COMM_WORLD.Send(&pMatrix(k % pMatrix.rows(), 0), pMatrix.cols(), MPI_DOUBLE, gMax.location / pMatrix.rows(), 1);
            for (int j = 0; j < pMatrix.cols(); j++) {
                pMatrix(k % pMatrix.rows(), j) = rowPivot[j];
            }
        }
    }
}

void invertParallel(Matrix& iA) {
    assert(iA.rows() == iA.cols());

    MPI_Barrier(COMM_WORLD);

    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    int lRank = COMM_WORLD.Get_rank();
    int lSize = COMM_WORLD.Get_size();

    mpi_double_int gMax;
    Matrix pMatrix(ceil(lAI.rows() / (float)lSize), lAI.cols());
    double rowPivot[pMatrix.cols()];

    SplitMatrix(pMatrix, lRank, lAI);

    for (size_t k = 0; k < iA.rows(); k++) {
        FindPivot(pMatrix, lRank, k, gMax);

        SwapAndBroadcastPivot(lRank, gMax, pMatrix, k, rowPivot);

        ApplyPivot(pMatrix, lRank, k, rowPivot);

        // cout << k << lRank << "\n" << pMatrix.str() << endl;
    }

    GatherResults(lRank, lAI, lSize, pMatrix, iA);
}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {

    // vérifier la compatibilité des matrices
    assert(iMat1.cols() == iMat2.rows());
    // effectuer le produit matriciel
    Matrix lRes(iMat1.rows(), iMat2.cols());
    // traiter chaque rangée
    for (size_t i = 0; i < lRes.rows(); ++i) {
        // traiter chaque colonne
        for (size_t j = 0; j < lRes.cols(); ++j) {
            lRes(i, j) = (iMat1.getRowCopy(i) * iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}

int main(int argc, char** argv) {

    MPI::Init();
    int lRank = COMM_WORLD.Get_rank();
    int lSize = COMM_WORLD.Get_size();
    int logMatrix = false;

    srand((unsigned)time(NULL));

    unsigned int lS = 5;
    if (argc == 2) {
        lS = atoi(argv[1]);
    }
    if (argc == 3) {
        logMatrix=(bool) argv[2];
    }

    MatrixRandom lA(lS, lS);
    Matrix lC(lA);
    Matrix lP(lA);

    double startSeq;
    double endSeq;
    double startPar;


    if (lRank == 0) {
        if (logMatrix){
            cout << "Matrice random:\n" << lA.str() << endl;
        }
        cout << "---Sequential Start" << endl;
        startSeq = MPI::Wtime();
        invertSequential(lC);
        endSeq =MPI::Wtime();
        Matrix lResSeq = multiplyMatrix(lA, lC);
        cout << "---Sequential End" << endl;

        cout << "Erreur Parallel : " << lResSeq.getDataArray().sum() - lS << endl;
        
    }
    MPI_Barrier(COMM_WORLD);
    if (lRank == 0) {
        cout << "\n---Parallel Start" << endl;
        startPar = MPI::Wtime();
    }

    invertParallel(lP);

    if (lRank == 0) {
        double endPar=MPI::Wtime();
        cout << "---Parallel End" << endl;
        if (logMatrix){
            cout << "Matrice inverse:\n" << lP.str() << endl;
        }

        Matrix lRes = multiplyMatrix(lA, lP);
        if (logMatrix){
            cout << "Produit des deux matrices:\n" << lRes.str() << endl;
        }
        cout << "Erreur Parallel : " << lRes.getDataArray().sum() - lS << endl;

        cout << "Time Sequential : "<< endSeq - startSeq << " , Time Parallel : "<<endPar - startPar<<endl;
    }

    MPI::Finalize();
    return 0;
}

