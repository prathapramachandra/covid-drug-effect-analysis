# COVID Drug Effect Analysis - Data Processing Pipeline
# Author: Prathap Ramachandra
# Description: Data cleaning, feature engineering and
# validation for COVID patient records analysis

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CovidDataProcessor:
    """
    Handles data cleaning, validation and feature
    engineering for COVID patient records
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.processed_df = None

    def load_data(self):
        """
        Load raw patient data from CSV
        """
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} patient records")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def validate_data(self):
        """
        Run data quality checks on raw data
        """
        logger.info("Running data validation checks...")

        total = len(self.df)

        # Check missing values
        missing = self.df.isnull().sum()
        missing_pct = (missing / total * 100).round(2)

        logger.info("Missing value report:")
        for col in self.df.columns:
            if missing[col] > 0:
                logger.warning(
                    f"  {col}: {missing[col]} missing "
                    f"({missing_pct[col]}%)"
                )

        # Check duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate records")

        # Check age range
        invalid_age = self.df[
            (self.df['age'] < 0) | (self.df['age'] > 120)
        ].shape[0]
        if invalid_age > 0:
            logger.warning(f"Found {invalid_age} invalid age values")

        logger.info("Data validation completed")
        return {
            'total_records': total,
            'missing_values': missing.to_dict(),
            'duplicates': duplicates,
            'invalid_ages': invalid_age
        }

    def clean_data(self):
        """
        Clean and standardize patient records
        """
        logger.info("Starting data cleaning...")
        df = self.df.copy()

        # Remove duplicates
        initial = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial - len(df)} duplicates")

        # Handle missing values
        df['age'].fillna(df['age'].median(), inplace=True)
        df['gender'].fillna('UNKNOWN', inplace=True)
        df['comorbidities'].fillna('NONE', inplace=True)
        df['recovery_days'].fillna(df['recovery_days'].median(), inplace=True)

        # Standardize text columns
        df['gender'] = df['gender'].str.upper().str.strip()
        df['treatment_type'] = df['treatment_type'].str.upper().str.strip()
        df['recovery_status'] = df['recovery_status'].str.upper().str.strip()

        # Remove invalid ages
        df = df[(df['age'] >= 0) & (df['age'] <= 120)]

        # Convert date columns
        df['treatment_start_date'] = pd.to_datetime(
            df['treatment_start_date']
        )
        df['treatment_end_date'] = pd.to_datetime(
            df['treatment_end_date']
        )

        logger.info(f"Cleaning complete. {len(df)} records remaining")
        self.processed_df = df
        return df

    def engineer_features(self):
        """
        Create new features for ML modeling
        """
        logger.info("Starting feature engineering...")
        df = self.processed_df.copy()

        # Age group feature
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 18, 35, 50, 65, 120],
            labels=['Under18', '18-35', '36-50', '51-65', 'Over65']
        )

        # Treatment duration
        df['treatment_duration'] = (
            df['treatment_end_date'] - df['treatment_start_date']
        ).dt.days

        # Comorbidity flag
        df['has_comorbidity'] = (
            df['comorbidities'] != 'NONE'
        ).astype(int)

        # Encode categorical variables
        df['gender_encoded'] = (df['gender'] == 'M').astype(int)

        df['outcome_label'] = (
            df['recovery_status'] == 'RECOVERED'
        ).astype(int)

        # Treatment encoding
        treatment_map = {
            t: i for i, t in enumerate(df['treatment_type'].unique())
        }
        df['treatment_encoded'] = df['treatment_type'].map(treatment_map)

        logger.info(f"Feature engineering complete. "
                   f"Created {len(df.columns)} features")
        self.processed_df = df
        return df

    def generate_summary_report(self):
        """
        Generate summary statistics report
        """
        df = self.processed_df

        report = {
            'total_patients': len(df),
            'treatment_distribution': df['treatment_type'].value_counts().to_dict(),
            'overall_recovery_rate': round(
                df['outcome_label'].mean() * 100, 2
            ),
            'avg_recovery_days': round(df['recovery_days'].mean(), 1),
            'age_statistics': {
                'mean': round(df['age'].mean(), 1),
                'median': df['age'].median(),
                'min': df['age'].min(),
                'max': df['age'].max()
            },
            'recovery_by_treatment': df.groupby('treatment_type')[
                'outcome_label'
            ].mean().round(4).to_dict()
        }

        logger.info("Summary Report:")
        logger.info(f"Total Patients: {report['total_patients']}")
        logger.info(f"Overall Recovery Rate: {report['overall_recovery_rate']}%")
        logger.info(f"Avg Recovery Days: {report['avg_recovery_days']}")

        return report

    def run_pipeline(self):
        """
        Run complete data processing pipeline
        """
        logger.info("Starting COVID data processing pipeline...")

        # Load data
        self.load_data()

        # Validate
        validation_report = self.validate_data()

        # Clean
        self.clean_data()

        # Engineer features
        self.engineer_features()

        # Generate report
        summary = self.generate_summary_report()

        logger.info("Pipeline completed successfully!")
        return {
            'validation': validation_report,
            'summary': summary,
            'processed_records': len(self.processed_df)
        }


if __name__ == "__main__":
    processor = CovidDataProcessor(
        data_path='data/covid_patient_records.csv'
    )
    result = processor.run_pipeline()
    print(f"Pipeline completed: {result}")
