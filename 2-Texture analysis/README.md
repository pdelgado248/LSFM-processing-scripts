The notebook featured in this folder (1-getTextureFeatures.ipynb) allows to read a series of 3D patches. These are normalized and the following texture features are extracted:

- LBP Uniformity
- Porosity 
- Mean Intensity
- Intensity Kurtosis
- Intensity Skewness
- Wavelet Vertical Detail Coefficient Mean
- Intensity Variance
- Wavelet Depth Detail Coefficient Mean
- Wavelet Horizontal Detail Coefficient Mean
- Total Energy in Frequency Domain
- Wavelet Approximation Coefficient Mean
- Fourier Dominant Frequency Magnitude

The feature vectors are then used to train an XGBoost classifier for healthy and pathological patches. Feature importance scores are obtained to rank these features per relevance for characterizing ADPKD.