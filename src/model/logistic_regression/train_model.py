    def upload_preprocessed_features(self):
        """Upload engineering dataset to Azure Data Lake"""
        # [TODO] Upload to a S3
        t_dir = cfg.TRANSFORMED_DIR
        if not os.path.exists(t_dir):
            os.mkdir(t_dir)

        df_eng = pl.concat([self.df_X, self.df_y], how="horizontal")
        df_eng.write_csv(f"{t_dir}/feats_engineered.csv")  

def train_model(self):
        """Balance training data using SMOTE"""
        X = self.df_X.to_numpy()
        y = self.df_y.to_numpy()

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, train_size=0.8, random_state=42
        )
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(X_train, y_train)
        self.model.fit(self.X_train, self.y_train)

    def save_test_data(self):
        """Save test data to CSV"""
        if not os.path.exists(cfg.TEST_DATA_DIR):
            os.mkdir(cfg.TEST_DATA_DIR)

        df_X = pl.DataFrame(self.X_test)
        df_y = pl.DataFrame(self.y_test)
        df_X.write_csv(f"{cfg.TEST_DATA_DIR}/X_test.csv")
        df_y.write_csv(f"{cfg.TEST_DATA_DIR}/y_test.csv")
        print(f"X_test, y_test saved to {cfg.TEST_DATA_DIR}")

    def save_model(self):
        """Train and save LR model"""
        pkl_path = f"{cfg.REGISTRY}/logistic_regression.pkl"
        joblib.dump(self.model, pkl_path)
        print(f"Model has been saved to {pkl_path}")