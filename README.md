# Partial Decoder Attention Network with Contour Weighted Loss Function for Data-Imbalance Medical Image Segmentation

### Abstract
Image segmentation is critically important in almost all medical image analysis for automatic interpretations and processing. However, it is often challenging to perform image segmentation due to data imbalance between intra- and inter-class, resulting in over- or under-segmentation. Consequently, we proposed a new methodology to address the above issue, with a compact yet effective contour-weighted loss function. Our new loss function incorporates a contour-weighted cross-entropy loss and separable dice loss. The former loss extracts the contour of target regions via morphological erosion and generates a weight map for the cross-entropy criterion, whereas the latter divides the target regions into contour and non-contour components through the extracted contour map, calculates dice loss separately, and combines them to update the network. We carried out abdominal organ segmentation and brain tumor segmentation on two public datasets to assess our approach. Experimental results demonstrated that our approach offered superior segmentation, as compared to several state-of-the-art methods, while in parallel improving the robustness of those popular state-of-the-art deep models through our new loss function.


### Conference paper
https://ieeexplore.ieee.org/abstract/document/10647384


### Contour-weighted loss
You can find the code of contour-weighted loss in the losses/BoundaryWeightedLoss.py.
