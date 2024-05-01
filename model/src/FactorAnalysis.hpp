/**
 * FactorAnalysis.hpp | bmi_model
 *
 * Description:
 * -----------
 * Class performing inference with and learning of a factor analysis model.
 *
 * Latent variable (vector): z ~ N(0, I)
 * Observation model: y = Cz + v, v ~ N(0, R)
 * y has size p
 * z has size k
 *
 * Note that the model assumes that the observations are zero-mean.
 *
 * C = factor loading matrix
 * R = diagonal matrix
 * Typically, dim(z) < dim(y).
 *
 * R and C are learned from data.
 * Then, model is inverted to obtain z given observation y.
 *
 * See for instance Roweis & Ghahramani, NECO (1999) for details and Ghahramani & Hinton's technical report (1996).
 */
#ifndef FactorAnalysis_hpp
#define FactorAnalysis_hpp

#include <Eigen/Dense>
#include <utility>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

class FactorAnalysis {
    Matrix C, R;
    Matrix beta;    // projection matrix used in inference
    Matrix V;       // covariance of the estimate
    Vector mean;    // mean of observations
    int k;          // number of factors
    int p;          // size of observation
    Matrix inverse_of_CCT_plus_R;
    
    
    ///Log-likehood evaluation.
    double loglikelihood(const Matrix& Y);
    
public:
    ///Ctor, with k the number of factors.
    FactorAnalysis(int k);
    
    /**
     * Infer the latent variable z using obervation y.
     *
     * Parameter:
     * ---------
     *  - y (Vector, size p): observation vector 
     *
     * Returns:
     * -------
     *  - mean (Vector, size k) or variance (Matrix, size k x k) of the posterior Gaussian given the observation, P(z | y)
     */
    Vector infer(const Vector& y);
    Matrix infer(const Matrix& Y);

    /**
     * Learn the R and C matrices.
     *
     * Parameter:
     * ---------
     * - data (Matrix, size p x n): n data points of size p, put in columns
     * - epsilon (double): convergence criterion
     */
    void learn(const Matrix& Y, double epsilon=1e-2);
    
    ///k-fold Cross-validation
    std::vector<double> cross_val(int nb_folds, int k_min, int k_max, const Matrix& Y);
    
    ///Getters
    Matrix get_C() {return C;}
    Matrix get_R() {return R;}
    Vector get_mean() {return mean;}
    Matrix get_beta() {return beta;}
    Matrix get_covariance() {return C*C.transpose() + R;}
    Matrix get_private_covariance() {return R;}
    Matrix get_shared_covariance() {return C*C.transpose();}
    int get_nb_factors() {return k;}
    int get_p() {return p;}
    
    ///Setters
    void set_beta(const Matrix& new_beta){beta = new_beta;}


};




#endif /* FactorAnalysis_hpp */
