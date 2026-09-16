#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "chroma.h"
#include "handle.h"
#include "io/cfgtype_io.h"
#include "util/gauge/gauge_startup.h"
#include "actions/ferm/fermacts/fermact_factory_w.h"
#include "actions/ferm/fermstates/ferm_createstate_aggregate_w.h"
#include "actions/ferm/fermstates/ferm_createstate_reader_w.h"
#include "actions/ferm/fermacts/clover_fermact_params_w.h"
#include "actions/ferm/fermacts/seoprec_clover_fermact_w.h"
#include "seoprec_wilstype_fermact_w.h"
#include "actions/ferm/linop/seoprec_clover_linop_w.h"
#include "qdp_util.h"

using namespace Chroma;

namespace
{
  // 输出 odd subset 上每个 site 的 24 个复数；spin 外层、color 内层。
  void dump_odd_fermion(const LatticeFermion &field, const std::string &path)
  {
    std::ofstream out(path, std::ios::trunc);
    out << std::setprecision(17);
    out << "# chroma_odd_fermion_v1\n";

    for (int site = 0; site < Layout::sitesOnNode(); ++site)
    {
      multi1d<int> coord = Layout::siteCoords(Layout::nodeNumber(), site);
      const int parity = (coord[0] + coord[1] + coord[2] + coord[3]) & 1;
      if (parity != 1)
        continue;

      Fermion value = peekSite(field, coord);
      out << coord[0] << " " << coord[1] << " " << coord[2] << " " << coord[3];
      for (int spin = 0; spin < Ns; ++spin)
      {
        ColorVector color_value = peekSpin(value, spin);
        for (int color = 0; color < Nc; ++color)
        {
          Complex z = peekColor(color_value, color);
          out << " " << toDouble(real(z)) << " " << toDouble(imag(z));
        }
      }
      out << "\n";
    }
  }

  bool linkageHack()
  {
    bool ok = true;
    ok &= CreateFermStateEnv::registerAll();
    ok &= WilsonTypeFermActs4DEnv::registerAll();
    ok &= InlineAggregateEnv::registerAll();
    ok &= GaugeInitEnv::registerAll();
    return ok;
  }
}

int main(int argc, char **argv)
{
  Chroma::initialize(&argc, &argv);
  QDPIO::cout << "linkage = " << linkageHack() << std::endl;

  const int nrow_in[4] = {12, 12, 12, 32};
  multi1d<int> nrow(4);
  nrow = nrow_in;
  Layout::setLattSize(nrow);
  Layout::create();

  Cfg_t cfg = {CFG_TYPE_SZINQIO,
               "/home/jyg/stout_smear/test_stout_smesr/2ml_1ms_lime/beta6.20_mu-0.2770_ms-0.2400_L12x32_cfg_1000.lime"};
  multi1d<LatticeColorMatrix> u(Nd);
  XMLReader gauge_file_xml, gauge_xml;
  gaugeStartup(gauge_file_xml, gauge_xml, u, cfg);

  // 与 PyQUDA/XML 相同的 4D stout + SEOPREC Clover FermState。
  const std::string fermact_xml = R"XML(
  <Param>
    <FermionAction>
      <FermAct>SEOPREC_CLOVER</FermAct>
      <Mass>-0.2770</Mass>
      <clovCoeff>0.15591192932912198</clovCoeff>
      <AnisoParam>
        <anisoP>false</anisoP>
        <t_dir>3</t_dir>
        <xi_0>1</xi_0>
        <nu>1</nu>
      </AnisoParam>
      <FermState>
        <Name>STOUT_FERM_STATE</Name>
        <rho>0.125</rho>
        <n_smear>1</n_smear>
        <FermionBC>
          <FermBC>SIMPLE_FERMBC</FermBC>
          <boundary>1 1 1 -1</boundary>
        </FermionBC>
      </FermState>
    </FermionAction>
  </Param>
  )XML";
  std::istringstream fermact_stream(fermact_xml);
  XMLReader fermact_reader(fermact_stream);

  Handle<FermState<LatticeFermion, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix>>> state;
  Handle<SymEvenOddPrecCloverLinOp> lin;
  try
  {
    using T = LatticeFermion;
    using P = multi1d<LatticeColorMatrix>;
    using Q = multi1d<LatticeColorMatrix>;
    Handle<CreateFermState<T, P, Q>> create_state(
        CreateFermStateEnv::reader(fermact_reader, "FermionAction"));
    CloverFermActParams clover_params(fermact_reader, "FermionAction");
    SymEvenOddPrecCloverFermAct fermact(create_state, clover_params);
    state = Handle<FermState<T, P, Q>>(fermact.createState(u));
    lin = Handle<SymEvenOddPrecCloverLinOp>(
        dynamic_cast<SymEvenOddPrecCloverLinOp *>(fermact.linOp(state)));
  }
  catch (const std::string &e)
  {
    QDPIO::cerr << "FermState/linOp error: " << e << std::endl;
    Chroma::finalize();
    return 1;
  }
  catch (const char *e)
  {
    QDPIO::cerr << "FermState/linOp error: " << e << std::endl;
    Chroma::finalize();
    return 1;
  }
  catch (const std::exception &e)
  {
    QDPIO::cerr << "FermState/linOp exception: " << e.what() << std::endl;
    Chroma::finalize();
    return 1;
  }

  // 使用与 PyQUDA ChromaRNG 相同的 seed 生成 odd-subset Gaussian 源。
  QDP::Seed seed;
  std::istringstream seed_xml(
      "<RNG><Seed><elem>1894</elem><elem>2</elem><elem>0</elem><elem>0</elem></Seed></RNG>");
  XMLReader seed_reader(seed_xml);
  read(seed_reader, "/RNG", seed);
  QDP::RNG::setrn(seed);

  LatticeFermion src = zero;
  gaussian(src, rb[1]);
  src *= std::sqrt(0.5);

  LatticeFermion out = zero;
  (*lin)(out, src, PLUS);

  const std::string prefix = "/home/jyg/stout_smear/PyQUDA_stout_smeared_force/PyQUDA/tests/hasenbusch_operation/rng_results/chroma_light_operator";
  dump_odd_fermion(src, prefix + "_src.txt");
  dump_odd_fermion(out, prefix + "_out.txt");
  QDPIO::cout << "src_norm2 = " << norm2(src, rb[1]) << std::endl;
  QDPIO::cout << "out_norm2 = " << norm2(out, rb[1]) << std::endl;

  Chroma::finalize();
  return 0;
}
