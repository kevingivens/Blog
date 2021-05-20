#include <iostream>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>
 
using namespace Eigen;
 
int main()
{
  
  ArrayXd S(21), C(21), Cn(21), delta(19), gamma(19), theta(19);
  
  // set up Spot grid
  S(0) = 0;
  double dx = 10;
 
  for (size_t i = 1; i < S.size(); i++)
  {
    S(i) = S(i-1) + dx;
  }

  std::cout << "S: " << S << std::endl;

  // simulation parameters
  double K = 100;
  double s = 0.2;
  double r = 0.05;
  int T = 1;
  int nx = 20;
  //double dx = 2*K/nx;
  int nt = (int)(T/(0.9/std::pow((s*nx),2))) + 1;
  double dt = T/nt;
  
  // Initial C at the Maturity
  for (size_t i = 0; i < C.size(); i++)
  {
    C(i) = std::max(S(i)-K, 0.0);
  }

  std::cout << "C: " << C << std::endl;
 

  //for (size_t i = 0; i < nt; i++)
  for (size_t i = 0; i < 4; i++)
  {
    delta = (0.5/dx)*(C.tail(19) - C.head(19));
    gamma = (1/std::pow(dx,2))*(C.tail(19) - 2*C.segment(1,19) + C.head(19));
    theta = -(0.5*std::pow(s,2))*square(S.segment(1,19))*gamma - r*S.segment(1,19)*delta + r*C.segment(1,19);
    std::cout << "delta at iteration: " << i << " " << delta << std::endl;
    std::cout << "gamma at iteration: " << i << " " << gamma << std::endl;
    std::cout << "theta at iteration: " << i << " " << theta << std::endl;
    std::cout << "C at iteration: " << i << " " << C << std::endl;
    Cn = std::move(C);
    std::cout << "C after move " << i << " " << C << std::endl;
    C.segment(1,19) = Cn.segment(1,19) - dt*theta;
    //spatial bc's
    C(0) = Cn(0)*(1 - r*dt);
    C(nx-1) = 2*C(nx-2) - C(nx-3);
    std::cout << "Cn at iteration: " << i << " " << Cn << std::endl;
    std::cout << "C final: " << i << " " << C << std::endl;
  }

  std::cout << "Cn: " << Cn << std::endl;
  std::cout << "C: " << C << std::endl;



}