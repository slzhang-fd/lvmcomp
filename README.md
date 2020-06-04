[![cran checks](https://cranchecks.info/badges/summary/lvmcomp)](https://cran.r-project.org/web/checks/check_results_lvmcomp.html)
[![R build status](https://github.com/slzhang-fd/lvmcomp/workflows/R-CMD-check/badge.svg)](https://github.com/slzhang-fd/lvmcomp/actions?workflow=R-CMD-check)
[![Travis-CI Build Status](https://travis-ci.com/slzhang-fd/lvmcomp.svg?branch=master)](https://travis-ci.com/slzhang-fd/lvmcomp)
[![CRAN\_Status\_Badge](http://www.r-pkg.org/badges/version/lvmcomp)](https://cran.r-project.org/package=lvmcomp)
[![downloads](http://cranlogs.r-pkg.org/badges/lvmcomp)](https://www.rdocumentation.org/trends)

# lvmcomp
A R package for analyzing large scale latent variable models with efficient parallel algorithms

# Description
Provides stochastic EM algorithms for latent variable models
with a high-dimensional latent space. So far, we provide functions for confirmatory item
factor analysis based on the multidimensional two parameter logistic (M2PL) model and the 
generalized multidimensional partial credit model. These functions scale well for problems
with many latent traits (e.g., thirty or even more) and are virtually tuning-free.
The computation is facilitated by multiprocessing 'OpenMP' API.
For more information, please refer to:
Zhang, S., Chen, Y., & Liu, Y. (2018). An Improved Stochastic EM Algorithm for Large-scale
Full-information Item Factor Analysis. British Journal of Mathematical and Statistical
Psychology. <doi:10.1111/bmsp.12153>.