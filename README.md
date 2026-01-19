# Early Detection of Cardiac Arrest in Newborn Babies

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning and deep learning-based system for early detection of cardiac arrest in newborn babies using physiological indicators and risk factors.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)
- [References](#references)
- [License](#license)

## 🎯 Overview

Cardiac arrest in newborn babies is an alarming medical emergency where early detection is critical for providing the best care and treatment. This project develops an advanced tool that integrates machine learning (Bagging Classifier) and deep learning (Neural Network) techniques to predict cardiac arrest in newborns based on various physiological factors and symptoms.

### Problem Statement

The symptoms of cardiac arrest in newborn babies can be subtle and difficult to detect, making early diagnosis and intervention challenging. This can lead to poor outcomes, including death or permanent brain damage.

### Solution

This tool assists healthcare professionals, particularly pediatricians, in swiftly and accurately predicting the onset of cardiac arrest in newborns by analyzing key health indicators.

## ✨ Features

- **Dual Model Approach**: Implements both Bagging Classifier and Deep Neural Network for robust predictions
- **User-Friendly GUI**: Built with PyQt5 for easy interaction
- **Real-time Predictions**: Instant risk assessment based on input parameters
- **High Accuracy**: Achieves reliable detection rates for early intervention
- **Comprehensive Analysis**: Evaluates multiple risk factors simultaneously

## 📊 Dataset

The model uses a dataset containing the following infant factors and symptoms:

| Feature | Description |
|---------|-------------|
| Birth Weight | Weight of the baby at birth |
| Family History | Genetic predisposition to cardiac issues |
| Preterm Birth | Whether the baby was born prematurely |
| Heart Rate | Current heart rate measurements |
| Breathing Difficulty | Respiratory distress indicators |
| Skin Tinge | Skin coloration abnormalities |
| Responsiveness | Baby's response to stimuli |
| Movement | Motor activity levels |
| Delivery Type | Mode of delivery (normal/cesarean) |
| Mother's BP History | Maternal blood pressure history |

## 🏗️ Architecture

The project comprises four main modules:

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Data Processing                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Preprocessing                      │
│            (Feature Scaling, Encoding, Splitting)            │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Module 1 & 2:         │     │   Module 3 & 4:         │
│   Bagging Classifier    │     │   Deep Neural Network   │
│   - Model Building      │     │   - Model Building      │
│   - Predictions         │     │   - Predictions         │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Results & Classification                  │
│              (Risk Assessment & Report Generation)           │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/cardiac-arrest-detection-newborn.git
   cd cardiac-arrest-detection-newborn
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Requirements

```txt
tensorflow>=2.0.0
keras>=2.4.0
scikit-learn>=0.24.0
pandas>=1.2.0
numpy>=1.19.0
PyQt5>=5.15.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

## 💻 Usage

### Running the Application

```bash
python main.py
```

### Using the GUI

1. Launch the application
2. Input the patient's physiological data
3. Select the prediction model (Bagging Classifier or Neural Network)
4. Click "Predict" to get the risk assessment
5. View the results and recommendations

### Command Line Interface

```bash
# Train the Bagging Classifier
python train_bagging.py --data dataset.csv

# Train the Neural Network
python train_neural_net.py --data dataset.csv --epochs 100

# Make predictions
python predict.py --model bagging --input patient_data.csv
```

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Bagging Classifier | 0.934 | 0.920 | 0.906 | 0.800 |
| Deep Neural Network | 0.921 | 0.878 | 0.892 | 0.809 |
| CMLA (Proposed) | 0.092 | 0.091 | 0.087 | 0.085 |

### Performance Metrics

The proposed CMLA model achieved:
- **Delta-P Value**: 0.09195
- **False Discovery Rate**: 0.09015
- **Prevalence Threshold**: 0.08665
- **Critical Success Index**: 0.08495

## 📁 Project Structure

```
cardiac-arrest-detection-newborn/
│
├── data/
│   ├── raw/                    # Raw dataset files
│   └── processed/              # Preprocessed data
│
├── models/
│   ├── bagging_classifier.pkl  # Trained Bagging model
│   └── neural_network.h5       # Trained Neural Network
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning and preparation
│   ├── bagging_model.py        # Bagging Classifier implementation
│   ├── neural_net_model.py     # Deep Neural Network implementation
│   ├── predict.py              # Prediction utilities
│   └── utils.py                # Helper functions
│
├── gui/
│   ├── main_window.py          # Main application window
│   ├── ui_components.py        # UI components
│   └── styles.qss              # Qt stylesheets
│
├── tests/
│   ├── test_models.py          # Model unit tests
│   └── test_preprocessing.py   # Preprocessing tests
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Data exploration
│
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # License file
```

## 🛠️ Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn (Bagging Classifier)
- **Deep Learning**: TensorFlow, Keras
- **GUI Framework**: PyQt5, PyUIC
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 👥 Contributors

| Name | Roll Number | Role |
|------|-------------|------|
| V. Bhavana | 20H51A0579 | Developer |
| S. Roshini | 20H51A05A4 | Developer |
| D. Sanjana | 20H51A0587 | Developer |

**Project Guide**: Ms. M.N. Sailaja, Assistant Professor, Dept. of CSE

**Institution**: CMR College of Engineering & Technology, Hyderabad

## 📚 References

1. Gupta, K., Jiwani, N., Pau, G., & Alibakhshikenari, M. - "A Machine Learning Approach Using Statistical Models for Early Detection of Cardiac Arrest in Newborn Babies in the Cardiac Intensive Care Unit" - IEEE Access

2. Choi, E., Schuetz, A., Stewart, W. F., & Sun, J. - "Using recurrent neural network models for early detection of heart failure onset" - J. Amer. Med. Inform. Assoc., vol. 24, no. 2, pp. 361–370

3. Rajkomar, A. et al. - "Scalable and accurate deep learning with electronic health records" - NPJ Digit. Med., vol. 1, no. 1, p. 18

4. [Python Documentation](https://www.python.org/)

5. [PyQt5 PyUIC](https://github.com/baoboa/pyqt5/blob/master/pyuic/uic/pyuic.py)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dr. Siva Skandha Sanagala, HOD, Dept. of CSE
- Dr. Vijaya Kumar Koppula, Dean-Academics
- Major Dr. V A Narayana, Principal
- CMR College of Engineering & Technology

---

<p align="center">
  <b>⚠️ Disclaimer</b>: This tool is intended for research and educational purposes only. Always consult qualified healthcare professionals for medical decisions.
</p>

<p align="center">
  Made with  at CMR College of Engineering & Technology
</p>
