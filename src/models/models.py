import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

class RetailReturnModeler:
    def __init__(self, config_path='../configs/params.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.params = yaml.safe_load(f)
        self.models = {}
        
    def load_and_split_data(self):
        """Tải dữ liệu đặc trưng và chia tập train/test"""
        # Load dữ liệu features đã được lưu từ bước 02
        df = pd.read_parquet(f"../{self.params['paths']['features_data']}")
        
        # Loại bỏ các cột ID không mang tính dự đoán
        cols_to_drop = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'is_return']
        X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        y = df['is_return']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.params['modeling']['test_size'], 
            random_state=self.params['random_seed'],
            stratify=y  # Cực kỳ quan trọng với dữ liệu mất cân bằng
        )
        return X_train, X_test, y_train, y_test

    def train_supervised_models(self, X_train, y_train):
        """Huấn luyện các mô hình phân lớp (Supervised Learning)"""
        # 1. Xử lý mất cân bằng bằng SMOTE nếu cấu hình yêu cầu
        if self.params['modeling']['smote']:
            print("Đang áp dụng SMOTE...")
            smote = SMOTE(random_state=self.params['random_seed'])
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train

        # 2. Baseline 1: Logistic Regression
        lr_params = self.params['modeling']['base_models']['logistic_regression']
        lr = LogisticRegression(C=lr_params['C'], max_iter=lr_params['max_iter'], random_state=self.params['random_seed'])
        lr.fit(X_train_res, y_train_res)
        self.models['Logistic_Regression'] = lr
        
        # 3. Baseline 2: Random Forest
        rf_params = self.params['modeling']['base_models']['random_forest']
        rf = RandomForestClassifier(n_estimators=rf_params['n_estimators'], max_depth=rf_params['max_depth'], random_state=self.params['random_seed'])
        rf.fit(X_train_res, y_train_res)
        self.models['Random_Forest'] = rf
        
        # 4. Advanced: XGBoost
        xgb_params = self.params['modeling']['advanced_model']['xgboost']
        xgb = XGBClassifier(
            n_estimators=xgb_params['n_estimators'], 
            learning_rate=xgb_params['learning_rate'], 
            random_state=self.params['random_seed'],
            eval_metric='logloss'
        )
        xgb.fit(X_train_res, y_train_res)
        self.models['XGBoost'] = xgb
        
        print("Đã huấn luyện xong các mô hình Supervised!")
        return self.models

    def train_semi_supervised(self, X_train, y_train, labeled_ratio=0.2):
        """Thực nghiệm Semi-Supervised: Giả lập ẩn nhãn (Yêu cầu để đạt điểm A)"""
        print(f"Bắt đầu thực nghiệm Semi-supervised với {labeled_ratio*100}% nhãn...")
        rng = np.random.RandomState(self.params['random_seed'])
        y_train_semi = np.copy(y_train)
        
        # Ẩn nhãn (gán = -1)
        unlabeled_indices = rng.rand(len(y_train)) > labeled_ratio
        y_train_semi[unlabeled_indices] = -1
        
        # Sử dụng Self-Training với base model là Random Forest
        rf_base = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.params['random_seed'])
        self_training_model = SelfTrainingClassifier(rf_base, threshold=0.8) # Ngưỡng tin cậy 80%
        
        self_training_model.fit(X_train, y_train_semi)
        self.models['Semi_Supervised_SelfTraining'] = self_training_model
        print("Đã huấn luyện xong mô hình Semi-supervised!")
        return self_training_model

    def evaluate_models(self, X_test, y_test):
        """Đánh giá và so sánh các mô hình"""
        results = []
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [np.nan]*len(y_test)
            
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if not np.isnan(y_proba[0]) else np.nan
            
            results.append({'Model': name, 'F1_Score': f1, 'ROC_AUC': auc})
            
            print(f"\n--- Báo cáo phân lớp: {name} ---")
            print(classification_report(y_test, y_pred))
            
        return pd.DataFrame(results).sort_values(by='F1_Score', ascending=False)