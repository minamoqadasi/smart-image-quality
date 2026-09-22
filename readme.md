# Smart Image Quality Assessment
This repository contains the code necessary for the Smart Image Quality Assessment project for COMP541: Data Mining.

The datasets being used for this project are the following:
- BIQ2021 Image Quality Assessment database
  - https://www.kaggle.com/datasets/nisarahmedrana/biq2021
- KonIQ-10k Image Quality Assessment database
  - https://database.mmsp-kn.de/koniq-10k-database.html
- LIVE In The Wild Image Quality Assessment database
  - https://live.ece.utexas.edu/research/ChallengeDB/

Dataset folder structure should be as follows:
```commandline
datasets
├── biq2021
│   ├── BIQ2021.csv
│   └── images
│       └── <images>
├── koniq
│   ├── images
│   │   └── <images>
│   ├── koniq10k_indicators.csv
│   └── koniq10k_scores_and_distributions.csv
└── livewild
    ├── AllImages_release.mat
    ├── AllMOS_release.mat
    ├── AllStdDev_release.mat
    ├── images
    │   └── <images>
    └── README.txt
```