# daibetisdetectionmodel
A Machinelearing Model that detects Dibetis of a paitent, using supervised learing XGBOOST CLASSIFIER
XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm based on boosted decision trees.
Boosting = combining multiple weak models (small decision trees) to create one strong model.
XGBoost improves accuracy by:
Building trees one after another
Each new tree fixes the errors of the previous trees
Uses gradient boosting (mathematics to reduce errors fast)
Has many optimization techniques to increase speed + accuracy

Why XGBoost is chosen for Diabetes Detection?
Diabetes detection is a classification problem → Output: Diabetic (1) or Not Diabetic (0).
XGBoost is a top choice because: 
works well with non-linear data i.e glucose level,insulin level, blood pressure,bmi etc
Autodetect missing value.

