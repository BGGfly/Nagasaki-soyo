<img width="274" height="80" alt="image" src="https://github.com/user-attachments/assets/6ff7e32c-fd76-4ebc-ba79-d04fa6a18165" />Intro
执行步骤按顺序：
问题一
              源域 (实验台数据)                         目标域 (列车实测数据)
        ┌─────────────────────────┐            ┌─────────────────────────┐
        │   data/origin/*.mat     │            │   data/target/*.mat     │
        └──────────┬──────────────┘            └──────────┬──────────────┘
                   │ (Step1) 筛选DE信号                        │ (StepX) 统一处理
                   ▼                                         ▼
        ┌─────────────────────────┐            ┌─────────────────────────┐
        │ step1_signals.npy       │            │   下采样 32kHz → 12kHz   │
        │ step1_labels.npy        │            │   小波降噪               │
        └──────────┬──────────────┘            │   标准化 (global_scaler) │
                   │ (Step2) 去噪+标准化        │  滑窗分片+特征提取        │
                   ▼                           └──────────┬──────────────┘
        ┌─────────────────────────┐                       ▼
        │ step2_processed_signals │            ┌─────────────────────────┐
        │ global_scaler.npz       │           │ target_features.npy      │
        └──────────┬──────────────┘            └─────────────────────────┘
                   │ (Step3) 特征提取
                   ▼
        ┌─────────────────────────┐
        │ step3_features.npy      │
        │ step3_labels.npy        │
        └─────────────────────────┘

        源域生成：
          step1_signals.npy：原始 DE 信号
          step1_labels.npy：对应标签 (0=N,1=OR,2=IR,3=B)
          step2_processed_signals.npy：去噪 + 标准化信号
          global_scaler.npz：标准化参数 (mean,std)
          step3_features.npy：13维特征矩阵
          step3_labels.npy：标签
        目标域生成：
          target_features.npy：经过下采样、去噪、标准化、分片、特征提取后的特征矩阵

        特征提取（13 维）
        每个分片提取 时域 / 频域 / 时频域特征：
        时域特征
          time_rms：均方根
          time_kurtosis：峭度
          time_skewness：偏度
          time_crest_factor：峰值因子（peak / rms）
          time_shape_factor：形状因子（rms / mean_abs）
          time_impulse_factor：冲击因子（peak / mean_abs）
        
        频域特征
          freq_centroid：频率中心
          freq_rms：频率 RMS
          freq_variance：频率方差
          freq_envelope_peak_freq：包络谱峰值频率
          
        时频域特征
          wp_entropy_low：小波包低频熵
          wp_entropy_mid：小波包中频熵
          wp_entropy_high：小波包高频熵
        共 13 个特征，每个分片得到一个 13 维向量



       频率归一化（RPM 标准化）
         假设源域信号采集时转速为 rpm_src，目标域转速为 rpm_target = 600，可做频率归一化：
          对频域特征归一化
          
          频率归一化后，源/目标域特征已经更接近
          可以用 DANN/CORAL/MMD 进行进一步对齐，消除剩余分布差异
      1.step1_read_data.py：
        数据读取与筛选遍历 data/origin/ 下所有 .mat 文件。只提取 驱动端 DE 信号（保证一致性）。
        文件名解析出 故障类型（N, OR, IR, B），并映射到标签：
        fault_type_map = {'N': 0, 'OR': 1, 'IR': 2, 'B': 3}
        保存：
          step1_signals.npy （原始信号列表，dtype=object，长度不一）
          step1_labels.npy （对应标签）
      2. step2_preprocess.py：
        输入：step1_signals.npy
        操作：
            小波降噪（db8，小波 5 层，软阈值处理）。
            全局标准化：把所有信号拼接 → 计算均值、标准差 → 每个信号 (x-mean)/std
        保存：
            step2_processed_signals.npy（降噪 + 标准化信号）
            global_scaler.npz（mean, std，用于目标域保持一致性）
      3.step3_extract_features.py：
        输入：step2_processed_signals.npy, step1_labels.npy
        操作：
            滑窗分片：窗口=2048，步长=1024
            提取 13维特征（时域6 + 频域4 + 时频3）：
            RMS、峭度、偏度、波峰因子、波形因子、冲击因子
            频谱质心、频谱RMS、频谱方差、包络谱峰值频率
            小波包熵（低频/中频/高频）
            清理 NaN/inf 样本。
          保存：
            step3_features.npy（形状：(样本数, 13)）
            step3_labels.npy（对应标签）
      4.stepX_preprocess_target.py:
        遍历目标域的 16 个 .mat 文件；
        下采样 32kHz → 12kHz；
        小波降噪；
        用之前保存的 global_scaler.npz （源域 Step2 得到的均值和标准差）做 全局标准化；
        滑窗分片（2048 窗口，1024 步长）提取 13维特征；
        保存到 target_features.npy，方便后续直接用迁移学习模型预测
问题二：
    1.划分训练集与测试集
            先将特征 X 和标签 y 加载
            按 70% 训练 / 30% 测试 划分
            对特征进行标准化（Z-score 或 MinMax），训练集用 fit_transform，测试集用 transform
    2.模型选择
        随机森林：
            对特征尺度不敏感
            容易训练
            可以得到特征重要性
    3.模型训练与评价指标
            准确率 (Accuracy)
            分类报告 (Precision, Recall, F1-score)
            混淆矩阵 (Confusion Matrix)
    task2_RF.py

问题三：
    1.迁移学习方法
            基于特征的迁移
                对源域和目标域的特征做对齐（比如归一化、频率归一化、标准化）
                使用PCA/t-SNE可视化确认特征分布是否接近
                训练源域分类器（任务2的模型），直接在目标域特征上预测（即“直接迁移”）
            基于模型的迁移
                在源域训练一个分类器（如 RandomForest、XGBoost 或 MLP）
                将训练好的模型在目标域做微调（fine-tune）
                微调可以用目标域少量已知标签（如部分标定数据）或者无标签的半监督方法
            领域自适应（Domain Adaptation）
                使用 DANN（Domain-Adversarial Neural Network） 或 CORAL 等方法
                在训练过程中强制源域和目标域特征分布相似，同时学习分类任务
                适合目标域完全无标签的场景
    2.数据处理与对齐
           StepX 处理中，已经：
            信号级去噪和标准化
            滑窗特征提取
            频率归一化
            特征级标准化（使用源域均值/标准差）
            所以源域和目标域特征现在已经在同一量纲下，可以直接输入到迁移模型。
    task3.py 模型迁移
    task4_predict_target.py 目标域预测




数据集：
    在工作目录里下的data文件夹下：
        origin：为源数据集 用于问题一的特征提取和问题二的模型训练
        CWRU_Bearing_Data：为额外补充的数据集，下载方式参考download.py用于在问题三中进行迁移学习时进行无监督学习
        target：目标数据集，在问题三中使用迁移的模型对其进行预测
      
