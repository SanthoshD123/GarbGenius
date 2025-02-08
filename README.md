# GARBGENIUS: AR APPROACH ENHANCING WARDROBE EXPERIENCE WITH RECOMMENDATION SYSTEM

## Overview
GarbGenius is an **AI-driven fashion recommendation system** that enhances the shopping experience with **Augmented Reality (AR)** and **machine learning algorithms**. It allows users to virtually try on clothes and receive personalized fashion recommendations based on their preferences and current trends. Initially, **K-Nearest Neighbors (KNN)** was used for recommendations, but **Support Vector Machine (SVM)** was later integrated to improve accuracy.
(I pulled recommendation system source code only, If you want AR code ask me at gmail : santhosh.damu78@gmail.com), Here publication for more detail [https://ijcrt.org/viewfull.php?&p_id=IJCRT24A4814]
## Features
- **AI-Powered Clothing Recommendations**
- **Virtual Try-On with Augmented Reality (AR)**
- **Vision Transformer (ViT) for Image Feature Extraction**
- **SVM for Enhanced Recommendation Accuracy**
- **Graphical User Interface (Tkinter-based)**

## Technologies Used
- **Python, Flask** (Backend Development)
- **Vision Transformer (ViT)** (Feature Extraction)
- **Support Vector Machine (SVM)** (Classification & Recommendations)
- **Pandas, NumPy** (Data Handling)
- **Tkinter** (GUI for User Interaction)
- **OpenCV & PIL** (Image Processing)
- **MeshLab** (3D Visualization)
- **Joblib** (Model Storage & Loading)

## Algorithmic Workflow

### 1. **Image Feature Extraction**
- Uses **Vision Transformer (ViT)** to extract clothing features from user-uploaded images.
- Converts images into numerical feature representations.

### 2. **Recommendation System**
- **KNN** was initially used but later replaced with **SVM**, which improved accuracy.
- The model is trained on labeled clothing datasets to predict recommendations.
- Uses **Cosine Similarity** to match user preferences.

### 3. **Virtual Try-On Experience**
- Integrates **Augmented Reality (AR)** to allow users to visualize how recommended outfits would look on them.
- Uses OpenCV and MeshLab for visualization.

## Performance & Accuracy

| Model | Accuracy |
|-----------------|----------|
| ResNet50 + KNN | 75% |
| ViT + Cosine Similarity | 77.9% |
| **ViT + SVM (Final Model)** | **80.8%** |

The **SVM-based approach** provided the highest accuracy, leading to its adoption.

## Future Improvements
- Expand dataset for better recommendations.
- Implement **Deep Learning-based virtual try-on**.
- Deploy a web-based interactive UI.

## License
This project is **MIT licensed**. Feel free to contribute!

## Contributing
Contributions are welcome! Open an issue or submit a pull request.

---
### Author
Santhosh D
Suryaa narayanan K
Thiyaneshwaran S
Vishal ponn rangan K

