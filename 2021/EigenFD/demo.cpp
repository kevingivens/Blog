#include <iostream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <Eigen/Dense>

using namespace std::chrono;
 
using namespace Eigen;
 
int main()
{

  auto start = high_resolution_clock::now();
  // C , Cn are the two price buffers used to propogate the Call value backwards in time
  // delta, gamma and theta are intermediate differential terms
  ArrayXd S(21), C(21), Cn(21), delta(19), gamma(19), theta(19);
  
  // set up spot grid
  S(0) = 0;
  double dx = 10;
 
  for (size_t i = 1; i < S.size(); i++)
  {
    S(i) = S(i-1) + dx;
  }

  // simulation parameters
  double K = 100;
  double s = 0.2;
  double r = 0.05;
  double T = 1;
  int nx = 20;
  int nt = (int)(T/(0.9/std::pow(s*nx,2))) + 1;
  double dt = T/nt;
  
  // Initialize C at the Maturity (boundary)
  for (size_t i = 0; i < C.size(); i++)
  {
    C(i) = std::max(S(i)-K, 0.0);
  }

  // Propagate system backwards in time, 
  // apply descretized diff eqs and spatial boundary conditions
  for (size_t i = 0; i < 10; i++)
  {
    delta = (0.5/dx)*(C.tail<19>() - C.head<19>());
    gamma = (1/std::pow(dx,2))*(C.tail<19>() - 2*C.segment<19>(1) + C.head<19>());
    theta = -(0.5*std::pow(s,2))*square(S.segment<19>(1))*gamma - r*S.segment<19>(1)*delta + r*C.segment<19>(1);
    // Move C into Cn 
    Cn = std::move(C);
    
    C.segment<19>(1) = Cn.segment<19>(1) - dt*theta;
    //spatial bc's
    C(0) = Cn(0)*(1 - r*dt);
    C(nx-1) = 2*C(nx-2) - C(nx-3);
    
  }
  
  // After function call
  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
  
  std::cout << "ran in: " << duration.count() << " microsecs "<< std::endl;

  std::cout << "C: " << C << std::endl;

}