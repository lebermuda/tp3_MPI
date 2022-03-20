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
    double rowPivot[pMatrix.cols()];

    //REPARTITION DES LIGNES ENTRE p PROCESSUS
    int u = 0;
    while (u*lSize+lRank<lAI.rows()){
        pMatrix.getRowSlice(u)=lAI.getRowCopy(u*lSize+lRank);
        u++;
    }
    // cout << lRank <<" :\n" << pMatrix.str() << endl;
    
    // traiter chaque rangée (répartition entre processus par i%lSize=lRang)
    for (size_t k=0; k<iA.rows(); k++) {
        // trouver l'index p du plus grand pivot de la colonne k en valeur absolue
        // (pour une meilleure stabilité numérique).
        
        // cout << "   COLONE " << k <<endl;
        
        //PIVOT MAX LOCAL
        mpi_double_int lMax;
        lMax.value=0;
        lMax.location = (int) k+lRank;

        for(size_t i = lMax.location/lSize; i < pMatrix.rows(); i++) {
            if(fabs(pMatrix(i,k)) > lMax.value) {
                lMax.value = fabs(pMatrix(i,k));
                lMax.location = (int) i*lSize+lRank;
            }
        }
        
        // PIVOT MAX GLOBAL 
        MPI_Allreduce(&lMax,&gMax,1,MPI_DOUBLE_INT,MPI_MAXLOC,COMM_WORLD);

        if(lRank==gMax.location%lSize){
            // cout << "pivot "<<k<<" " <<gMax.location<<" : "<<gMax.value << endl;
            // vérifier que la matrice n'est pas singulière
            if (pMatrix(gMax.location/lSize, k) == 0) throw runtime_error("Matrix not invertible");
            // échanger la ligne courante avec celle du pivot
            double lValue = pMatrix(gMax.location/lSize,k);
            for (int j=0;j<pMatrix.cols();j++){
                pMatrix(gMax.location/lSize,j)/=lValue;
                rowPivot[j]=pMatrix(gMax.location/lSize,j);
            }
            MPI_Bcast(&rowPivot, pMatrix.cols(), MPI_DOUBLE, gMax.location%lSize, COMM_WORLD);
            // cout<<k<<lRank<< " Envoie [ "<<rowPivot[0]<<", "<<rowPivot[1]<<", "<<rowPivot[2]<<", "<<rowPivot[3]<<", "<<rowPivot[4]<<", "<<rowPivot[5]<<", "<<rowPivot[6]<<", "<<rowPivot[7]<<", "<<rowPivot[8]<<" ]"<<endl;

            //SWAP de ligne k et pivot max
            if (gMax.location != k) {
                if (k%lSize!=lRank){
                    Status lStatus;
                    double swapRow[pMatrix.cols()];
                    COMM_WORLD.Recv(&swapRow, pMatrix.cols(), MPI_DOUBLE, k%lSize, 1, lStatus);
                    for (int j=0;j<pMatrix.cols();j++){
                        pMatrix(gMax.location/lSize,j)=swapRow[j];
                    }
                    // cout << pMatrix.str() <<endl;
                }
                else{
                    pMatrix.swapRows(gMax.location/lSize, k/lSize);                   
                }
            }
        }
        else{
            MPI_Bcast(&rowPivot, pMatrix.cols(), MPI_DOUBLE, gMax.location%lSize, COMM_WORLD);
            
            //SWAP de ligne k et pivot max
            if(lRank==k%lSize){
                COMM_WORLD.Send(&pMatrix(k,0),pMatrix.cols(),MPI_DOUBLE,gMax.location%lSize,1);
                for (int j=0;j<pMatrix.cols();j++){
                    pMatrix(k/lSize,j)=rowPivot[j];
                }
            }
        }

        MPI_Barrier(COMM_WORLD);

        // Pour chaque rangée...
        int i = 0;
        while (i*lSize+lRank<lAI.rows()){
            if (i*lSize+lRank != k) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
                double lValue = pMatrix(i, k);

                for (int j=0;j<pMatrix.cols();j++){
                    if(j<k){
                        cout <<"k="<<k<<" , pMatrix(i="<<i<<",j="<<j<<") : "<< pMatrix(i,j) <<" ,rowPivot(j) : "<< rowPivot[j]<<" , lValue " << lValue << endl;
                    }
                    pMatrix(i,j) -= rowPivot[j]*lValue;

                }
            }
            i++;
        }

        // MPI_Barrier(COMM_WORLD);
        // cout << k << lRank<<" : \n" << pMatrix.str()<<endl;
        // MPI_Barrier(COMM_WORLD);
        
    }
    
    u = 0;
    while (u*lSize+lRank<lAI.rows()){
        for (int j=0;j<iA.cols();j++){
            iA(u*lSize+lRank,j)=pMatrix(u,iA.cols()+j);
        }
        u++;
    }
    

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

