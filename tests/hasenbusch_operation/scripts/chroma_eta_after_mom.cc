#include "chroma.h"
#include "qdp_util.h"
#include <cmath>
#include <iostream>

using namespace Chroma;

int main(int argc, char **argv)
{
  Chroma::initialize(&argc, &argv);
  multi1d<int> nrow(4);
  nrow[0] = 12;
  nrow[1] = 12;
  nrow[2] = 12;
  nrow[3] = 32;
  Layout::setLattSize(nrow);
  Layout::create();

  QDP::Seed seed;
  std::istringstream seed_xml(
      "<RNG><Seed><elem>2094</elem><elem>850</elem><elem>1</elem><elem>0</elem></Seed></RNG>");
  XMLReader seed_reader(seed_xml);
  read(seed_reader, "/RNG", seed);
  QDP::RNG::setrn(seed);

  multi1d<LatticeColorMatrix> p(Nd);
  for (int mu = 0; mu < Nd; ++mu) {
    gaussian(p[mu]);
    p[mu] *= sqrt(Real(0.5));
    taproj(p[mu]);
  }

  LatticeFermion eta = zero;
  gaussian(eta, rb[1]);
  eta *= sqrt(Real(0.5));

  QDPIO::cout << "eta_norm2 = " << norm2(eta, rb[1]) << std::endl;
  Chroma::finalize();
  return 0;
}
