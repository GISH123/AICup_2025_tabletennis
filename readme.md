# AI CUP 2025 春季賽 – 桌球智慧球拍資料的精準分析  
TEAM_7697 競賽解法

本repo提供 TEAM_7697 在 AI CUP 2025「桌球智慧球拍資料的精準分析競賽」之完整程式碼與重現步驟。  
核心以 **Python 3.10 + LightGBM / XGBoost / CatBoost** 進行梯度提升樹模型訓練，透過 27 段等長揮拍特徵 + 多模型加權融合，Private Leaderboard 成績 **0.80052 (Rank 26)**。

---

## 目錄結構

AICup_2025_tabletennis/  
├
├─ 20250601_o3_new_holdout_gish_0.85495208.ipynb # 主訓練 + 推論腳本  
├─ 競賽報告與程式碼TEAM_7697桌球智慧球拍資料的精準分析競賽.docx # 第二階段競賽文件報告  
├─ 39_Training_Dataset/ # 官方訓練資料（自行下載放置）  
├─ 39_Test_Dataset/ # 官方測試資料（自行下載放置）  
├─ requirements.txt  
└─ README.md  

---

# 1. 下載專案
git clone https://github.com/GISH123/AICup_2025_tabletennis.git

# 2. 建立並啟用 Conda 環境
conda create -n aicup2025 python=3.10 -y
conda activate aicup2025

# 3. 安裝依賴
pip install -r requirements.txt
GPU 使用者：LightGBM 4.6.0 wheel 已支援 CUDA。若想啟用 GPU，請確定已安裝對應版本的 CUDA 與 cuDNN。


39_Training_Dataset/  
   ├─ train_info.csv  
   └─ train_data/*.txt  
39_Test_Dataset/  
   ├─ test_info.csv  
   └─ test_data/*.txt  

20250601_o3_new_holdout_gish_0.85495208.ipynb 產生檔案  
submission_GISH_blended.csv # 機率加權融合（官方上傳）
submission_GISH_rank.csv # Rank 平均融合（比較用）

主要方法
固定 27 段揮拍切分 → Swing 時域 / 頻域 / 小波域混合特徵

Session 級統計 + 分位數聚合 (mean / std / max / min / q10~q90)

LightGBM + CatBoost + XGBoost 以 5-fold Group K-Fold (player_id) 交叉驗證

簡易權重搜尋 模型融合  

Private leaderboard AUC = 0.80052

License
僅限 AI CUP 2025 競賽學術交流使用，禁止未經授權之商業用途。
Copyright © 2025 TEAM_7697
