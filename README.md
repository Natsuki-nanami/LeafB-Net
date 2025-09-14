# Title
LeafB-Net: Integrating Frequency and Spatial Features with Dual-Channel Attention for Robust Tomato Disease Diagnosis
# Problem Description
Tomato is an important economic crop, yet it is severely affected and destroyed by diseases. Existing deep learning-based methods for tomato leaf disease recognition suffer from significant issues, including high computational cost, suboptimal recognition performance, and limited application scenarios. 
# Proposed Study
This model explicitly integrates frequency domain analysis with spatial feature extraction to accurately capture subtle disease characteristics with extremely low computational overhead. The first three sections of the network consist of an efficient and lightweight module called the LeafB block (LB block), which incorporates discrete wavelet transform (DWT). The DWT can directly and efficiently capture high-frequency features like edges and textures from the frequency domain. These features are crucial for distinguishing diseases but are often overlooked in standard spatial convolutions. Furthermore, we innovatively designed a Dual-channel Efficient Channel Attention (DualECA) module based on global average and max pooling information to further amplify critical features and significantly enhance the model's sensitivity to key lesion characteristics. In the deeper layers of the network, we introduce a Triple Attention Block (TA Block) to leverage its dimensional interaction capabilities, refine high-level semantics, and further improve the model's discriminative power while reducing the number of parameters.Evaluation experiments were conducted on a high-quality tomato leaf disease dataset.
# Overall architecture of LeafB-Net.
<img width="747" height="404" alt="image" src="https://github.com/user-attachments/assets/357af32a-97da-4ab3-b311-ad102df386e3" />

# Result
The results demonstrate that LeafB-Net achieves an accuracy of 97.88% and an F1-score of 97.12% with only 5.30M parameters and 256 MFLOPs. Its performance surpasses existing deep learning classification methods. This study provides an efficient, reliable, and interpretable technical solution for the rapid diagnosis of crop diseases in mobile applications within the field of precision agriculture.
# Repo Structure
The LeafB-Net method has the following structure:


# Install Dependencies
```
!pip install -r requirement.txt
```
# Github Cloning
