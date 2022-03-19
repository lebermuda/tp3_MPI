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
    for (size_t k=0; k<iA.rows(); ++k) {
        // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
        // (pour une meilleure stabilité numérique).
        size_t p = k;
        double lMax = fabs(lAI(k,k));
        for(size_t i = k; i < lAI.rows(); ++i) {
            if(fabs(lAI(i,k)) > lMax) {
                lMax = fabs(lAI(i,k));
                p = i;
            }
        }
        // vérifier que la matrice n'est pas singulière
        if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");

        // échanger la ligne courante avec celle du pivot
        if (p != k) lAI.swapRows(p, k);

        double lValue = lAI(k, k);
        for (size_t j=0; j<lAI.cols(); ++j) {
            // On divise les éléments de la rangée k
            // par la valeur du pivot.
            // Ainsi, lAI(k,k) deviendra égal à 1.
            lAI(k, j) /= lValue;
        }

        // Pour chaque rangée...
        for (size_t i=0; i<lAI.rows(); ++i) {
            if (i != k) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
                double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
            }
        }
    }

    // On copie la partie droite de la matrice AI ainsi transformée
    // dans la matrice courante (this).
    for (unsigned int i=0; i<iA.rows(); ++i) {
        iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
    }
}

// Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
void invertParallel(Matrix& iA) {
    // // vous devez coder cette fonction
    // MPI::Init();

    // int lSize = MPI::COMM_WORLD.Get_size();
    // int lRank = MPI::COMM_WORLD.Get_rank();

    // float tab[5] = {1.,2.,3.,4.,5.};
    // float lMax=fabs(tab[lRank]);
    // int lPivot=lRank;
    // int gPivot;

    // for (int k=lRank; k<5; k+=lSize) {
    //     if (lMax<fabs(tab[k])){
    //         lPivot=k;
    //         lMax=fabs(tab[k]);
    //     }
    // }

    // cout << lRank << " : " <<lPivot <<endl;

    // MPI_Reduce(&lPivot,&gPivot,1,MPI_INT,MPI_MAX,0,COMM_WORLD);

    // if (lRank==0){
    //     cout << gPivot << " => " << tab[gPivot] << endl;
    // }

    // MPI::Finalize();

    //DEBUT DE REFLEXION
    //peut etre fait p fois ?
    // vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
    // construire la matrice [A I]
    MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));

    int lRank = COMM_WORLD.Get_rank();
    int lSize = COMM_WORLD.Get_size();
    
    mpi_double_int gMax;
    Matrix pMatrix(lAI.rows()/lSize+1,lAI.cols());

    int k = 0;
    while (k*lSize+lRank<lAI.rows()){
        pMatrix.getRowSlice(k)=lAI.getRowCopy(k*lSize+lRank);
        k++;
    }
    // cout << lRank <<" :\n" << pMatrix.str() << endl;
    
    // traiter chaque rangée (répartition entre processus par i%lSize=lRang)
    for (size_t k=0; k<iA.rows(); k++) {
        // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
        // (pour une meilleure stabilité numérique).
        
        // cout << "   COLONE " << k <<endl;

        mpi_double_int lMax;
        lMax.value=0;
        lMax.location = (int) k+lRank;

        for(size_t i = lMax.location/lSize; i < pMatrix.rows(); i++) {
            if(fabs(pMatrix(i,k)) > lMax.value) {
                lMax.value = fabs(pMatrix(i,k));
                lMax.location = (int) i*lSize+lRank;
            }

            //  cout << "Ligne "<<i<<" par p="<<lRank<<endl;
        }
        
        // REDUCTION (f=Max) DES p DE CHAQUE PROCESSUS 
        MPI_Allreduce(&lMax,&gMax,1,MPI_DOUBLE_INT,MPI_MAXLOC,COMM_WORLD);

        //FAIRE LA VERIFICATION QU'UNE FOIS
        if(lRank==gMax.location%lSize){
            cout << "pivot "<<k<<" " <<gMax.location<<" : "<<gMax.value << endl;
            // vérifier que la matrice n'est pas singulière
            if (lAI(gMax.location, k) == 0) throw runtime_error("Matrix not invertible");
            // échanger la ligne courante avec celle du pivot
            if (gMax.location != k) {
                if (gMax.location%lSize!=lRank%lSize){
                    //SENDRECV
                }
                else{
                    pMatrix.swapRows(gMax.location/lSize, k/lSize);
                }
            }

            double lValue = lAI(k, k);
            for (size_t j=0; j<lAI.cols(); ++j) {
                // On divise les éléments de la rangée k
                // par la valeur du pivot.
                // Ainsi, lAI(k,k) deviendra égal à 1.
                lAI(k, j) /= lValue;
            }
        }
        MPI_Barrier(COMM_WORLD);

        //BROADCAST p
        //MPI_Bcast(&lAI(gMax.location,0), lAI.rows(), MPI_DOUBLE, gMax.location%lSize, COMM_WORLD);

        // Pour chaque rangée...
        for (size_t i=lRank; i<lAI.rows(); i+=lSize) {
            if (i != k) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
                double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
            }
        }

        MPI_Barrier(COMM_WORLD);
        
    }


    // On copie la partie droite de la matrice AI ainsi transformée
    // dans la matrice courante (this).
    for (unsigned int i=lRank; i<iA.rows(); i+=lSize) {
        iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
    }
        // On copie la partie droite de la matrice AI ainsi transformée
    // dans la matrice courante (this).
    // if (lRank==0){
    //     for (unsigned int i=0; i<iA.rows(); ++i) {
    //         iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
    //     }
    // }
    

}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {

    // vérifier la compatibilité des matrices
    assert(iMat1.cols() == iMat2.rows());
    // effectuer le produit matriciel
    Matrix lRes(iMat1.rows(), iMat2.cols());
    // traiter chaque rangée
    for(size_t i=0; i < lRes.rows(); ++i) {
        // traiter chaque colonne
        for(size_t j=0; j < lRes.cols(); ++j) {
            lRes(i,j) = (iMat1.getRowCopy(i)*iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}

int main(int argc, char** argv) {
    
    MPI::Init();
    int lRank = COMM_WORLD.Get_rank();
    int lSize = COMM_WORLD.Get_size();

    srand((unsigned)time(NULL));
    
    unsigned int lS = 5;
    if (argc == 2) {
        lS = atoi(argv[1]);
    }

    MatrixRandom lA(lS, lS);
    
    if (lRank==0){
        cout << "Matrice random:\n" << lA.str() << endl;
    }

    // Matrix lB(lA);
    // invertSequential(lB);
    // cout << "Matrice inverse:\n" << lB.str() << endl;

    // Matrix lRes = multiplyMatrix(lA, lB);
    // cout << "Produit des deux matrices:\n" << lRes.str() << endl;

    // cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;

    Matrix lC(lA);
    invertParallel(lC);

    if (lRank==0){
        cout << "Matrice inverse:\n" << lC.str() << endl;

        Matrix lRes = multiplyMatrix(lA, lC);
        cout << "Produit des deux matrices:\n" << lRes.str() << endl;

        cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;
    }
    
    MPI::Finalize();
    return 0;
}

