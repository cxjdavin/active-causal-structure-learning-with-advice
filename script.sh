# Download and setup Julia
wget https://julialang-s3.julialang.org/bin/mac/aarch64/1.8/julia-1.8.3-macaarch64.tar.gz
tar zxvf julia-1.8.3-macaarch64.tar.gz
./julia-1.8.3/bin/julia -e 'import Pkg; Pkg.add("PyCall"); Pkg.add("LightGraphs"); Pkg.add("LinkedLists")'

# Grab PADS source files
wget -r --no-parent --no-host-directories --cut-dirs=1 http://www.ics.uci.edu/\~eppstein/PADS/

# Clone CliquePicking repository
git clone https://github.com/mwien/CliquePicking.git

# Copy chordal graphs from CliquePicking repository
for i in {16,32,64}; do mkdir wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/subtree-logn/subtree-n=${i}-logn-nr=1.gr wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/subtree-2logn/subtree-n=${i}-2logn-nr=1.gr wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/subtree-sqrtn/subtree-n=${i}-sqrtn-nr=1.gr wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/interval/interval-n=${i}-nr=1.gr wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/peo-2/peo-n=${i}-2-nr=1.gr wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/peo-4/peo-n=${i}-4-nr=1.gr wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/thickening-3/thickening-n=${i}-3-nr=1.gr wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/thickening-logn/thickening-n=${i}-logn-nr=1.gr wbl_chordal_${i}; done
for i in {16,32,64}; do cp CliquePicking/aaai_experiments/instances/thickening-sqrtn/thickening-n=${i}-sqrtn-nr=1.gr wbl_chordal_${i}; done

python3 -m venv venv
source venv/bin/activate
pip install wheel
pip install causaldag==0.1a135
pip install tqdm p_tqdm

# Run experiments to obtain plots
python3 experiments_wbl.py 16
python3 experiments_wbl.py 32
python3 experiments_wbl.py 64

