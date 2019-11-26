
** updates
## 26 Nov, 2019
- replace old adaptive rejection sampling (ARS) code with adaptive rejection metropolis hastings sampling 
  algorithm (ARMS) code by Wally Gilks, which is more robust and efficient.
- add new function `SAEM_mirt()` of MIRT estimation with stochastic approximation (Polyak-Rupper Averaging)
- add new function `SOE_mirt()` which employs a special prior so that $theta$ has posterior normal distribution

## 19 Dec, 2018
- fix cran comments: provide toy examples to be quickly executed by user in less than 5 sec for each Rd file
- small modification to StEM_mirt() function: set minimum burn-in step size to be TT=20 by default
- small modification to StEM_pcirt() function: set minimum burn-in step size to be TT=20 by default
- fix Step output bug: Step start from 1 instead of 0
- update version number to 1.2

## 11 Oct, 2018
- update Github repository link
- update Travis-CI icon
- add BugReport link