/**
 * FactorAnalysis.cpp | bmi_model
 */
#include "FactorAnalysis.hpp"
#include <iostream>
#include <fstream>
#include <Eigen/LU>
#include <Eigen/SVD>
#include "rand_mat.hpp"

FactorAnalysis::FactorAnalysis(int k) : k(k) {}

Vector FactorAnalysis::infer(const Vector& y){
    return beta * (y - mean);
}

Matrix FactorAnalysis::infer(const Matrix& Y){
    Matrix Y_minus_mean(Y.rows(), Y.cols());
    Y_minus_mean = Y;
    Y_minus_mean.colwise() -= mean;
    return beta * Y_minus_mean;
}


void FactorAnalysis::learn(const Matrix& Y, double epsilon) {
    double ll_prev{ 0. }, ll{ 0. };
    p = int(Y.rows());
    const int n{ int(Y.cols()) }; // number of data points
    
    // Compute sample mean and covariance
    mean = Vector::Zero(p);
    mean = Y.rowwise().mean();
    Matrix Y_minus_mean(Y.rows(), Y.cols());
    Y_minus_mean = Y;
    Y_minus_mean.colwise() -= mean;
    
    Matrix S = 1./double(n) * Y_minus_mean * Y_minus_mean.transpose();
    
    // Compute eigenval/vec
    Eigen::SelfAdjointEigenSolver<Matrix> eigensolver(S);
    if (eigensolver.info() != Eigen::Success) abort();
    Matrix EigenVectors = eigensolver.eigenvectors();
    Vector EigenValues = eigensolver.eigenvalues();
    
    // Compute number of dimensions to get to 0.99 x total variance
    double total_variance{ 0. };
    for (int i=0; i<EigenValues.size(); i++) total_variance += EigenValues(i);
    
    double cumulative_variance{ 0. };
    int count{ 1 };
    while (cumulative_variance < 0.99*total_variance){
        cumulative_variance += EigenValues(EigenValues.size() - count);
        count++;
    }
    std::cout << "Number of dimensions for at least 0.99 of total variance = " << count << std::endl;

    //Eigen::BDCSVD<Matrix> svd_dec(S);
    //Vector s_vals = svd_dec.singularValues();
    
    // Initialize C and R
    C = Matrix::Zero(p, k); //random_gaussian_matrix(p, k, 1./sqrt(double(k)));
    for (int c=0;c<k;c++) C.col(c) = sqrt(EigenValues(p-1-c)) * EigenVectors.col(p-1-c);
    
    R = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(p);
    R.diagonal() = Vector::Ones(p); //s_vals;
    
    ll = loglikelihood(Y);
    int ll_iter{ 0 };
    do {
        ll_prev = ll;
        
        // Compute beta
        beta = C.transpose() * inverse_of_CCT_plus_R;
        
        // Compute V
        V = Matrix::Identity(k,k) - beta * C;
        
        // Infer latent variable
        Matrix Z = infer(Y);
        
        // Update factor loading and private noise matrices
        Matrix delta = Y_minus_mean * Z.transpose();
        Matrix gamma = Z * Z.transpose() + n*V;
        
        C = delta * gamma.partialPivLu().inverse();
        Vector diag = (S - C * delta.transpose()/double(n)).diagonal();
        R.diagonal() = diag.array().abs(); // Very important!! To make sure that the diagonal of R are > 0
        
        //Evaluate log-likelihood
        ll = loglikelihood(Y);
        ll_iter++;
        if (ll_iter % 1000 == 0) std::cout << ll << '\n';
    
    } while (ll - ll_prev > epsilon);
    //std::cout << "Final log-likelihood = " << ll << std::endl;
}


double FactorAnalysis::loglikelihood(const Matrix& Y){
    Matrix Y_minus_mean(Y.rows(), Y.cols());
    Y_minus_mean = Y;
    Y_minus_mean.colwise() -= mean;
    
    double ll = -Y_minus_mean.cols() * log((C*C.transpose() + R).partialPivLu().determinant());
    
    Matrix Rinv = Matrix::Zero(p, p);
    for (int i=0; i<p; i++){
        Rinv(i, i) = 1./R.diagonal()(i);
    }
    inverse_of_CCT_plus_R = Rinv - Rinv * C * (Matrix::Identity(k, k) + C.transpose() * Rinv * C).partialPivLu().inverse()*C.transpose()*Rinv;
    
    for (int c{0}; c<Y_minus_mean.cols(); c++){
        ll -= Y_minus_mean.col(c).transpose() * inverse_of_CCT_plus_R * Y_minus_mean.col(c);
    }
    
    ll -= Y_minus_mean.cols()*Y_minus_mean.rows()*log(2.*M_PI); //not really necessary, but useful for comparison with sklearn
    
    return 0.5 * ll;
}

std::vector<double> FactorAnalysis::cross_val(int nb_folds, int k_min, int k_max, const Matrix& Y){
    std::vector<double> mean_loglikelihoods(k_max - k_min + 1, 0.);
    
    const int nb_samples{ int(Y.cols()) };
    const int test_size{ nb_samples / nb_folds };
    
    const int k_original{ k };
    
    //Construct folds
    std::vector<int> v(nb_samples);
    std::iota(v.begin(), v.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(v.begin(), v.end(), g);
    
    std::vector<Matrix> training_folds(nb_folds, Matrix::Zero(Y.rows(), nb_samples - test_size));
    std::vector<Matrix> test_folds(nb_folds, Matrix::Zero(Y.rows(), test_size));
    
    
    for(int f=0; f<nb_folds; f++){
        //training_folds[f] = Matrix(Y.rows(), nb_samples - test_size);
        //test_folds[f] = Matrix(Y.rows(), test_size);
        
        int count_training{ 0 };
        int count_test{ 0 };
        
        for (int i=0; i<f*test_size; i++) {
            training_folds[f].col(count_training) = Y.col(v[i]);
            count_training++;
        }
        for (int i=f*test_size; i<(f+1)*test_size; i++){
            test_folds[f].col(count_test) = Y.col(v[i]);
            count_test++;
        }
        for (int i=(f+1)*test_size; i<nb_samples; i++){
            training_folds[f].col(count_training) = Y.col(v[i]);
            count_training++;
        }
        //std::cout << "Fold " << f + 1 << " training data\n";
        //std::cout << training_folds[f].row(0) << std::endl;
        //std::cout << "\nFold " << f + 1 << " test data\n";
        //std::cout << test_folds[f].row(0) << std::endl;
    }
    
    std::ofstream outfile("training_fold.txt");
    for (int r=0; r<training_folds[0].rows(); r++){
        for (int c=0; c<training_folds[0].cols(); c++) {
            outfile << training_folds[0](r, c) << '\t';
        }
        outfile << '\n';
    }
    outfile.close();
    
    // Cross-validate
    std::cout << "FA cross-validation\n";
    for (int nb_factors=k_min; nb_factors <= k_max; nb_factors++){
        k = nb_factors;
        for (int f=0; f<nb_folds; f++){
            learn(training_folds[f]);
            mean_loglikelihoods[nb_factors-k_min] += loglikelihood(test_folds[f]);
        }
        mean_loglikelihoods[nb_factors-k_min] /= nb_folds;
        std::cout<<nb_factors << ": " << "mean logLL = " << mean_loglikelihoods[nb_factors-k_min] << '\n';
    }

    // Relearn using the original k
    k = k_original;
    learn(Y);
    
    return mean_loglikelihoods;
}
