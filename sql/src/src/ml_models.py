# COVID Drug Effect Analysis - Machine Learning Models
# Author: Prathap Ramachandra
# Description: Comparing multiple ML models to predict
# COVID-19 treatment outcomes using causal inference

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreatmentOutcomePredictor:
    """
    Compares multiple ML models to predict COVID-19
    treatment outcomes and identify best performing model
    """

    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        }
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'age',
            'gender_encoded',
            'age_group_encoded',
            'treatment_encoded',
            'dosage',
            'treatment_duration',
            'has_comorbidity',
            'hospital_stay_days'
        ]

    def prepare_data(self, df):
        """
        Prepare features and target for modeling
        """
        logger.info("Preparing data for modeling...")

        # Select features
        X = df[self.feature_columns].copy()
        y = df['outcome_label'].copy()

        # Handle any remaining missing values
        X = X.fillna(X.median())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Training set: {len(X_train)} records")
        logger.info(f"Test set: {len(X_test)} records")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all ML models
        """
        logger.info("Training and evaluating models...")

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")

            # Train model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Cross validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=5, scoring='accuracy'
            )

            self.results[model_name] = {
                'accuracy': round(accuracy * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'cv_mean': round(cv_scores.mean() * 100, 2),
                'cv_std': round(cv_scores.std() * 100, 2)
            }

            logger.info(f"{model_name} Results:")
            logger.info(f"  Accuracy: {accuracy * 100:.2f}%")
            logger.info(f"  Precision: {precision * 100:.2f}%")
            logger.info(f"  Recall: {recall * 100:.2f}%")
            logger.info(f"  F1 Score: {f1 * 100:.2f}%")
            logger.info(
                f"  CV Score: {cv_scores.mean() * 100:.2f}% "
                f"(+/- {cv_scores.std() * 100:.2f}%)"
            )

        return self.results

    def get_best_model(self):
        """
        Identify best performing model based on accuracy
        """
        best = max(
            self.results,
            key=lambda x: self.results[x]['accuracy']
        )
        self.best_model = best

        logger.info(f"Best performing model: {best}")
        logger.info(f"Accuracy: {self.results[best]['accuracy']}%")

        return best, self.results[best]

    def generate_comparison_report(self):
        """
        Generate model comparison report
        """
        logger.info("\n" + "="*60)
        logger.info("MODEL COMPARISON REPORT")
        logger.info("="*60)

        report_df = pd.DataFrame(self.results).T
        report_df = report_df.sort_values('accuracy', ascending=False)

        logger.info("\n" + report_df.to_string())

        logger.info("\n" + "="*60)
        logger.info(f"BEST MODEL: {self.best_model}")
        logger.info(
            f"ACCURACY: {self.results[self.best_model]['accuracy']}%"
        )
        logger.info("="*60)

        return report_df

    def analyze_treatment_effectiveness(self, df):
        """
        Analyze treatment effectiveness across patient groups
        """
        logger.info("Analyzing treatment effectiveness...")

        # Recovery rate by treatment
        treatment_analysis = df.groupby('treatment_type').agg(
            total_patients=('patient_id', 'count'),
            recovered=('outcome_label', 'sum'),
            avg_recovery_days=('recovery_days', 'mean'),
            avg_hospital_stay=('hospital_stay_days', 'mean')
        ).reset_index()

        treatment_analysis['recovery_rate'] = round(
            treatment_analysis['recovered'] /
            treatment_analysis['total_patients'] * 100,
            2
        )

        treatment_analysis = treatment_analysis.sort_values(
            'recovery_rate',
            ascending=False
        )

        logger.info("\nTreatment Effectiveness Analysis:")
        logger.info(treatment_analysis.to_string())

        # Age group analysis
        age_analysis = df.groupby(['age_group', 'treatment_type']).agg(
            total_patients=('patient_id', 'count'),
            recovery_rate=('outcome_label', 'mean'),
            avg_recovery_days=('recovery_days', 'mean')
        ).reset_index()

        age_analysis['recovery_rate'] = round(
            age_analysis['recovery_rate'] * 100,
            2
        )

        return {
            'treatment_analysis': treatment_analysis,
            'age_group_analysis': age_analysis
        }

    def run_full_analysis(self, df):
        """
        Run complete ML analysis pipeline
        """
        logger.info("Starting full ML analysis pipeline...")

        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Train and evaluate models
        self.train_and_evaluate(X_train, X_test, y_train, y_test)

        # Get best model
        best_model, best_results = self.get_best_model()

        # Generate comparison report
        comparison_report = self.generate_comparison_report()

        # Analyze treatment effectiveness
        effectiveness = self.analyze_treatment_effectiveness(df)

        logger.info("Full ML analysis pipeline completed!")

        return {
            'model_results': self.results,
            'best_model': best_model,
            'best_results': best_results,
            'comparison_report': comparison_report,
            'effectiveness_analysis': effectiveness
        }


if __name__ == "__main__":
    from data_processing import CovidDataProcessor

    # Load and process data
    processor = CovidDataProcessor(
        data_path='data/covid_patient_records.csv'
    )
    processor.run_pipeline()
    df = processor.processed_df

    # Run ML analysis
    predictor = TreatmentOutcomePredictor()
    results = predictor.run_full_analysis(df)

    print(f"\nBest Model: {results['best_model']}")
    print(f"Accuracy: {results['best_results']['accuracy']}%")
