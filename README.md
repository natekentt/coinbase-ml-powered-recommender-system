# coinbase-ml-powered-recommender-system
This repository contains the foundational components for a high-performance, real-time cryptocurrency and asset recommendation engine. The system is designed using a Dual-Tower Neural Network architecture, which separates user and item feature encoding for efficient retrieval from a Vector Database.


Creating trainig env macOS:

conda create -n final_tf2 python=3.10 -c conda-forge -y
conda activate final_tf2

conda install -c conda-forge \
    tensorflow=2.12 \
    pandas \
    scikit-learn \
    numpy=1.23 \
    protobuf=3.20 -y

# IMPORTANT: Install the Apple Metal bridge separately using pip, 
# as it bypasses the AVX instruction issue and forces GPU acceleration.
pip install tensorflow-metal==0.8.0