# Biquality Learning for Distribution Shift

This repository provides a reference implementation of the algorithm described in the paper :

[Biquality Learning: a Framework to Design Algorithms Dealing with Closed-Set Distribution Shifts]()

## Replication

In order to run the experiments and generate results files and figures, run the following lines :

```
git clone https://github.com/pierrenodet/blds.git
cd blds
tar -xf datasets.tar.gz
python3 benchmark.py output/
python3 benchmark_cluster.py output_cluster/
python3 post_process.py output/results.csv output/stats.csv figures/
python3 post_process_cluster.py output_cluster/results.csv output_cluster/stats.csv figures_cluster/
```