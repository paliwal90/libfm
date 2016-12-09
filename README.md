libFM with Updated Early Stopping and others
=========================

Changes:

* Extended early stop updates learning rate (divide by 2) after each early stop activation. However early stop is deactivated if learn rate reaches lower than 1e-6. From then, iterations continues with lastly updated learn rate.

* Intermediate predictions saving at any desired iteration.

* SGDA added.

* Load model functionality added (useful for starting from previously saved model coefficients).

* AUC evaluate per iteration. If early stop is used, validation auc can be maximised. Earlier validation log loss was minimized. Now have 2 options.

* Makefile in src is updated (std c++) to support lambda expressions in ranking function for auc computation.


Additional parameters:

* optimize_metric auc -- which metric to optimize on validation set by early stop (two allowed: logloss and auc, default: logloss). This option used only when early stop is used.

* pred_iter_step -- set iteration step at which to save intermidiate predictions. E.g. if set to 10, then after every 10th iteration, predictions will be saved (output files will be generated with informative naming).

* load_model -- saved_model_path


Example
=======

``` bash
<path-to-libfm>/bin/libFM -task c -train <path-to-train-data> -test <path-to-test-data>
-validation <path-to-validation-data> -dim '1,1,8' -early_stop 1 -num_stop 15 -optimize_metric auc 
-pred_iter_step iter_step -out <where-predictions-to-save> -verbosity 0 -iter 40 
-method sgd  -learn_rate 0.001 -init_stdev 0.0003 -load_model saved_model_path
```


libFM
=====

Library for factorization machines

web: http://www.libfm.org/

forum: https://groups.google.com/forum/#!forum/libfm

Factorization machines (FM) are a generic approach that allows to mimic most factorization models by feature engineering. This way, factorization machines combine the generality of feature engineering with the superiority of factorization models in estimating interactions between categorical variables of large domain. libFM is a software implementation for factorization machines that features stochastic gradient descent (SGD) and alternating least squares (ALS) optimization as well as Bayesian inference using Markov Chain Monte Carlo (MCMC).

Compile
=======
libFM has been tested with the GNU compiler collection and GNU make. libFM and the tools can be compiled with
> make all

Usage
=====
Please see the [libFM 1.4.2 manual](http://www.libfm.org/libfm-1.42.manual.pdf) for details about how to use libFM. If you have questions, please visit the [forum](https://groups.google.com/forum/#!forum/libfm).
