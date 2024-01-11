# BPA-Alternatives-AR

Implementation of the paper "Screening of the Antagonistic Activity of Potential Bisphenol A Alternatives Towards the Androgen Receptor Using Machine Learning and Molecular Dynamics Simulation" by Zeguo Yang, Ling Wang, Ying Yang, Xudi Pang, Yuzhen Sun, Yong Liang, Huiming Cao\*

---

![model_framework](./process_diagram.png)

---
This code is based on the following works：  
  - [DGL-LifeSci: An Open-Source Toolkit for Deep Learning on Graphs in Life Science](https://pubs.acs.org/doi/10.1021/acsomega.1c04017)
  * [D-MPNN: Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237)
  - Cao's work [Investigation of the Binding Fraction of PFAS in Human Plasma and Underlying Mechanisms Based on Machine Learning and Molecular Dynamics Simulation](https://pubs.acs.org/doi/10.1021/acs.est.2c04400) 
#### 0-Install-script : Install dependencies　　
    We have provided relevant installation script examples, please refer to them and make changes according to your environment.
- [DMPNN_install_script.txt](0-Install-script/DMPNN_install_script.txt)
* [DGL-lifesci_install_script.txt](0-Install-script/DGL-lifesci_install_script.txt)
- [Pycaret_install_script.txt](0-Install-script/Pycaret_install_script.txt)
### Instruction
#### 1-GNNs：This includes all the source code of AFP,GAT,GCN,MPNN in this study.
- **In AFP GAT GCN.ipynb**.
 ```
    # Adjust the parameters here, and refer to the cell above for the description of the parameters
        GPUNum = '1'
        repetitions = 10
        seed = 0 
        args = parser.parse_args(args=[......]).__dict__
 ```
#### 1-ML-based：This includes all the source code of ML-based models in this study.
#### 1-Pycaret：This includes all the source code of EVS dataset trained using pycaret that only contains three or ten descriptors in this study.
#### 2-DMPNN&hyD-MPNN：This includes all the source code and feature data of DMPNN and hyD-MPNN in this study.
- **5_AR_6108_NURA_chemprop.csv**：NuRA-AR’s SMILES dataset.
* **Features**：The fusion features used in hyD-MPNN include the total alva descriptors dataset and the dataset divided into 10 seeds.
- **Metric_Chemporp.ipynb**：A tool script for summarizing prediction results.
* **Running script examples**：
 ```
# D-MPNN
chemprop_hyperopt --dataset_type classification --num_iters 300 --data_path 5_AR_6108_NURA_chemprop.csv --config_save_path Config/config_AR_6108.json
chemprop_train --data_path 5_AR_6108_NURA_chemprop.csv  --dataset_type classification --split_sizes 0.8 0.1 0.1  --save_dir Checkpoints/AR_6108 --config_path Config/config_AR_6108.json --metric accuracy  --epochs 300 --extra_metrics  auc f1 mcc --gpu 0  --num_folds 10 --seed 1
chemprop_predict --test_path predata.csv --individual_ensemble_predictions --checkpoint_dir Checkpoints/AR_6108 --preds_path Predict/AR_6108_Predict_Result.csv

# hyD-MPNN(seed1)
chemprop_hyperopt --dataset_type classification --num_iters 100 --data_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_train_1_smi.csv --separate_val_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_val_1_smi.csv --separate_test_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_test_1_smi.csv --config_save_path Config/config_AR_6108_NURA_slim_Normalize_hype_1.json --features_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_train_1_features.csv --separate_val_features_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_val_1_features.csv --separate_test_features_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_test_1_features.csv --no_features_scaling --gpu 0
chemprop_train --data_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_train_1_smi.csv --separate_val_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_val_1_smi.csv --separate_test_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_test_1_smi.csv --dataset_type classification --save_dir Checkpoints/AR_6108_NURA_slim_Normalize/DMPNN_alva_slim_Normalize_hype_1 --config_path Config/config_AR_6108_NURA_slim_Normalize_hype_1.json --metric accuracy --epochs 300 --extra_metrics auc f1 mcc --gpu 0 --features_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_train_1_features.csv --separate_val_features_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_val_1_features.csv --separate_test_features_path AR_alva_6108_slim_Normalize_smi_feature/AR_6108_NURA_slim_Normalize_test_1_features.csv --no_features_scaling
chemprop_predict --test_path predata.csv --no_features_scaling --checkpoint_dir Checkpoints/AR_6108_NURA_slim_Normalize/DMPNN_alva_slim_Normalize_hype_1 --preds_path Predict/AR_6108_NURA_slim_Normalize/DMPNN_alva_slim_Normalize_hype_1_Predict_Result.csv --features_path predata_features.csv
 ``` 
#### 2-HyML-based：This includes all the source code of HyML-based models in this study.
* **model**：Hybrid models and Hyperparameters.
- **1-AR-Alva-6108-slim-Normalize-group.csv**：The hybrid features of DMPNN and alva, with alva undergoing RFE selection..
* **output**：Prediction results of NuRA-AR test set.
- **In hyLGB hySVM hyXGB.ipynb**.
 ```
    # Adjust the parameters here
    tasks_dic = {'1-AR-AlvaSlim-DMPNN-6108-Normalize-group.csv': ['activity']}
    file_name = '1-AR-AlvaSlim-DMPNN-6108-Normalize-group.csv'
    task_type = 'cla'  # 'reg' or 'cla'
    dataset_label = file_name.split('/')[-1].split('_')[0]
    tasks = tasks_dic[dataset_label]
    OPT_ITERS = 50
    repetitions = 10
    num_pools = 10
    unbalance = True
    patience = 100
    ecfp = True
    space_ = {'num_leaves': hp.choice('num_leaves', range(2,256,1)),
              'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
              'n_estimators': hp.choice('n_estimators', range(10,300,1)),
              'min_split_gain': hp.uniform('min_split_gain', 0., 1),
              'reg_alpha': hp.loguniform('reg_alpha', 1e-10, 10.0),
              'reg_lambda': hp.loguniform('reg_lambda', 1e-10, 10.0),
              'feature_fraction': hp.uniform('feature_fraction', 0.4, 1),
              'bagging_fraction': hp.uniform('bagging_fraction', 0.4, 1),
              'bagging_freq': hp.choice('bagging_freq', range(0,7,1)),
              'min_child_samples': hp.choice('min_child_samples', range(1,100,1))
              }
    num_leaves_ls = range(2,256,1)
    n_estimators_ls = range(10,300,1)
    bagging_freq_ls = range(0,7,1)
    dataset = pd.read_csv(file_name)
    pd_res = []

 ``` 
#### 3-docking files：This includes the RF scores of five molecular docking calculations for the compounds in APS in this study.



