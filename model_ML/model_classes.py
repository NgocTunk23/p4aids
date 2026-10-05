import time
from sklearn.metrics import recall_score, f1_score, log_loss, brier_score_loss, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

class BaseAnomalyModel:
    def __init__(self, name, model_class, default_params, **custom_params):
        """
        Khởi tạo mô hình cơ sở.
        :param name: Tên mô hình (vd: 'XGBoost')
        :param model_class: Class mô hình của thư viện (vd: XGBClassifier)
        :param default_params: Bộ tham số mặc định
        :param custom_params: Bộ siêu tham số (Hyperparameters) do người dùng truyền vào để đè lên mặc định
        """
        self.name = name
        self.params = default_params.copy()
        self.params.update(custom_params) # Cho phép người dùng ghi đè siêu tham số
        self.model = model_class(**self.params)
        
    def update_params(self, **new_params):
        """Cập nhật siêu tham số sau khi khởi tạo"""
        self.params.update(new_params)
        self.model.set_params(**new_params)
        print(f"[{self.name}] Đã cập nhật siêu tham số: {new_params}")
        
    def train(self, X_train, y_train):
        """Huấn luyện mô hình và đo thời gian"""
        print(f"[{self.name}] Đang huấn luyện với tham số: {self.params}")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.train_time = time.time() - start_time
        return self.train_time
        
    def predict(self, X_val):
        return self.model.predict(X_val)
        
    def predict_proba(self, X_val):
        return self.model.predict_proba(X_val)[:, 1]
        
    def evaluate(self, X_val, y_val, print_report=False):
        """Đánh giá toàn diện mô hình trên tập Validation"""
        from sklearn.metrics import recall_score, f1_score, log_loss, brier_score_loss, classification_report, fbeta_score
        y_pred = self.predict(X_val)
        y_prob = self.predict_proba(X_val)
        
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        f2 = fbeta_score(y_val, y_pred, beta=2.0)
        loss = log_loss(y_val, y_prob)
        brier = brier_score_loss(y_val, y_prob)
        
        custom_score = f2 - (0.1 * loss)
        
        metrics = {
            "Model": self.name,
            "Custom Score": round(custom_score, 4),
            "F2-Score": round(f2, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Log Loss": round(loss, 4),
            "Brier Score": round(brier, 4),
            "Time (s)": round(getattr(self, 'train_time', 0), 2)
        }
        
        if print_report:
            print(f"\n[{self.name}] BÁO CÁO KẾT QUẢ CHI TIẾT:")
            print(classification_report(y_val, y_pred))
            
        return metrics

# ==========================================
# CÁC CLASS MÔ HÌNH CỤ THỂ DÀNH CHO USER
# ==========================================

class XGBoostModel(BaseAnomalyModel):
    def __init__(self, **kwargs):
        default_params = {
            'random_state': 42, 
            'n_estimators': 100, 
            'learning_rate': 0.1, 
            'n_jobs': -1, 
            'eval_metric': 'logloss'
        }
        super().__init__("XGBoost", XGBClassifier, default_params, **kwargs)

class LightGBMModel(BaseAnomalyModel):
    def __init__(self, **kwargs):
        default_params = {
            'random_state': 42, 
            'n_estimators': 100, 
            'learning_rate': 0.1, 
            'n_jobs': -1, 
            'verbose': -1
        }
        super().__init__("LightGBM", LGBMClassifier, default_params, **kwargs)

class CatBoostModel(BaseAnomalyModel):
    def __init__(self, **kwargs):
        default_params = {
            'random_state': 42, 
            'iterations': 100, 
            'learning_rate': 0.1, 
            'thread_count': -1, 
            'verbose': 0
        }
        super().__init__("CatBoost", CatBoostClassifier, default_params, **kwargs)

class RandomForestModel(BaseAnomalyModel):
    def __init__(self, **kwargs):
        default_params = {
            'random_state': 42, 
            'n_estimators': 100, 
            'n_jobs': -1
        }
        # Random Forest thường chạy rất chậm. Khuyến nghị giảm n_estimators hoặc giới hạn max_depth
        super().__init__("Random Forest", RandomForestClassifier, default_params, **kwargs)

class ExtraTreesModel(BaseAnomalyModel):
    def __init__(self, **kwargs):
        default_params = {
            'random_state': 42, 
            'n_estimators': 100, 
            'n_jobs': -1
        }
        super().__init__("Extra Trees", ExtraTreesClassifier, default_params, **kwargs)
