/// Chroma RNG 对拍 dump 工具（只读新增，不改动 chroma 本体）。
///
/// 用法：chroma_rng_dump <Lx> <Ly> <Lz> <Lt> <seed> <mode> <out.txt> [precision]
///   mode: uniforms | eta | mom
///   precision: double | single（仅记录在头部；实际精度由 chroma build 决定）
///
/// 输出格式与 pyquda.chroma_rng.ChromaRNG.dump_reference 相同：
///   首行 "# {...json...}"；随后每行 "x y z t <值...>"
///   uniforms: 24 float（random 场，float32 值）
///   eta:      24 double（gaussian 输出，spin 外层 / color 内层 / (re,im)）
///   mom:      72 double（mu=0..3 依次 3x3 复矩阵 re,im 交错；已 x sqrt(0.5)、taproj）
///
/// 说明：单条 trajectory 的 RNG 消费顺序中，动量 gaussian 先于 eta；
/// 本工具一次只跑一种 mode（每次独立 setrn），对拍时 pyquda 侧同样独立取流。

#include "chroma.h"
#include "actions/ferm/fermacts/fermact_factory_w.h"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>

using namespace QDP;
using namespace Chroma;

int main(int argc, char* argv[])
{
  if (argc < 8) {
    std::cerr << "usage: " << argv[0]
              << " <Lx> <Ly> <Lz> <Lt> <seed> <uniforms|eta|mom> <out.txt> [precision]\n";
    return 1;
  }

  const int Lx = std::atoi(argv[1]);
  const int Ly = std::atoi(argv[2]);
  const int Lz = std::atoi(argv[3]);
  const int Lt = std::atoi(argv[4]);
  const unsigned long seed_in = std::strtoul(argv[5], nullptr, 10);
  const std::string mode = argv[6];
  const std::string out_path = argv[7];
  const std::string prec = (argc > 8) ? argv[8] : "unknown";

  Chroma::initialize(&argc, &argv);

  {
    multi1d<int> nrow(Nd);
    nrow[0] = Lx;
    nrow[1] = Ly;
    nrow[2] = Lz;
    nrow[3] = Lt;
    Layout::setLattSize(nrow);
    Layout::create();
  }

  // seed → QDP Seed（12-bit 字组装，与 PSeed::assign 一致）
  Seed rng_seed;
  rng_seed = static_cast<int>(seed_in & 0xffffffffUL);
  RNG::setrn(rng_seed);

  multi1d<int> bcs(Nd);
  bcs[0] = bcs[1] = bcs[2] = 1;
  bcs[3] = -1;
  Handle<FermBC<LatticeFermion, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> > > fbc(
      new SimpleFermBC<LatticeFermion, multi1d<LatticeColorMatrix>, multi1d<LatticeColorMatrix> >(bcs));

  const std::string rank_path = out_path + ".rank" + std::to_string(Layout::nodeNumber());
  std::ofstream fout(rank_path.c_str());
  fout << std::setprecision(17);
  {
    std::ostringstream hdr;
    hdr << "{\"format\":\"chroma_rng_dump_v1\",\"latt\":[" << Lx << "," << Ly << "," << Lz << ","
        << Lt << "],\"seed\":" << seed_in << ",\"mode\":\"" << mode << "\",\"precision\":\"" << prec
        << "\",\"source\":\"chroma\"}";
    fout << "# " << hdr.str() << "\n";
  }

  const Subset& odd = rb[1];

  if (mode == "uniforms" || mode == "eta") {
    LatticeFermion field;
    field = zero;
    if (mode == "uniforms") {
      random(field, odd);          // 原始均匀数（float32 值）
    } else {
      gaussian(field, odd);        // Box-Muller 输出
    }

    const int* tab = odd.siteTable().slice();
    for (int j = 0; j < odd.numSiteTable(); ++j) {
      const int site = tab[j];
      const multi1d<int> c = Layout::siteCoords(Layout::nodeNumber(), site);
      const auto fs = peekSite(field, c).elem();
      if (mode == "uniforms") {
        fout << c[0] << " " << c[1] << " " << c[2] << " " << c[3];
        for (int k = 0; k < 24; ++k)
        {
          const int s = k / (Nc * 2);
          const int col = (k / 2) % Nc;
          const int r = k % 2;
          fout << " " << toDouble(r == 0 ? fs.elem(s).elem(col).real() : fs.elem(s).elem(col).imag());
        }
        fout << "\n";
      } else {
        // 输出顺序：spin 外层、color 内层、(re,im)，与 pyquda 侧一致
        fout << c[0] << " " << c[1] << " " << c[2] << " " << c[3];
        for (int s = 0; s < Ns; ++s)
          for (int col = 0; col < Nc; ++col)
            for (int r = 0; r < 2; ++r)
              fout << " " << toDouble(r == 0 ? fs.elem(s).elem(col).real() : fs.elem(s).elem(col).imag());
        fout << "\n";
      }
    }
  } else if (mode == "mom") {
    multi1d<LatticeColorMatrix> p(Nd);
    for (int mu = 0; mu < Nd; ++mu) {
      p[mu] = zero;
      gaussian(p[mu]);
      p[mu] *= sqrt(Real(0.5));
      taproj(p[mu]);
    }

    const int* tab = all.siteTable().slice();
    for (int j = 0; j < all.numSiteTable(); ++j) {
      const int site = tab[j];
      const multi1d<int> c = Layout::siteCoords(Layout::nodeNumber(), site);
      fout << c[0] << " " << c[1] << " " << c[2] << " " << c[3];
      for (int mu = 0; mu < Nd; ++mu) {
        const auto fs = peekSite(p[mu], c).elem();
        for (int i = 0; i < Nc * Nc; ++i)
          for (int r = 0; r < 2; ++r)
          {
            const int row = i / Nc;
            const int col = i % Nc;
            fout << " " << toDouble(r == 0 ? fs.elem().elem(row, col).real() : fs.elem().elem(row, col).imag());
          }
      }
      fout << "\n";
    }
  } else {
    std::cerr << "unknown mode: " << mode << "\n";
    Chroma::finalize();
    return 1;
  }

  fout.close();
  Chroma::finalize();
  return 0;
}
