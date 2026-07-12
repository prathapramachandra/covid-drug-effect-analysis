-- COVID Drug Effect Analysis - Data Extraction Queries
-- Author: Prathap Ramachandra
-- Description: SQL queries for extracting and transforming
-- clinical patient data for COVID treatment analysis

-- ============================================
-- 1. Extract Raw Patient Records
-- ============================================
SELECT
    p.patient_id,
    p.age,
    p.gender,
    p.blood_type,
    p.comorbidities,
    t.treatment_type,
    t.treatment_start_date,
    t.treatment_end_date,
    t.dosage,
    o.recovery_status,
    o.recovery_days,
    o.hospital_stay_days,
    o.outcome_date
FROM patients p
JOIN treatments t ON p.patient_id = t.patient_id
JOIN outcomes o ON p.patient_id = o.patient_id
WHERE t.treatment_start_date >= '2020-01-01'
AND t.treatment_start_date <= '2022-12-31'
ORDER BY p.patient_id;

-- ============================================
-- 2. Data Quality Validation Check
-- ============================================
SELECT
    COUNT(*) AS total_records,
    COUNT(CASE WHEN patient_id IS NULL THEN 1 END) AS missing_patient_id,
    COUNT(CASE WHEN age IS NULL THEN 1 END) AS missing_age,
    COUNT(CASE WHEN gender IS NULL THEN 1 END) AS missing_gender,
    COUNT(CASE WHEN treatment_type IS NULL THEN 1 END) AS missing_treatment,
    COUNT(CASE WHEN recovery_status IS NULL THEN 1 END) AS missing_outcome,
    COUNT(CASE WHEN recovery_days IS NULL THEN 1 END) AS missing_recovery_days,
    ROUND(
        COUNT(CASE WHEN recovery_status IS NOT NULL THEN 1 END) * 100.0 / COUNT(*),
        2
    ) AS data_completeness_percentage
FROM patients p
JOIN treatments t ON p.patient_id = t.patient_id
JOIN outcomes o ON p.patient_id = o.patient_id;

-- ============================================
-- 3. Treatment Distribution Analysis
-- ============================================
SELECT
    treatment_type,
    COUNT(*) AS total_patients,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage,
    AVG(recovery_days) AS avg_recovery_days,
    MIN(recovery_days) AS min_recovery_days,
    MAX(recovery_days) AS max_recovery_days,
    COUNT(CASE WHEN recovery_status = 'RECOVERED' THEN 1 END) AS recovered_count,
    ROUND(
        COUNT(CASE WHEN recovery_status = 'RECOVERED' THEN 1 END) * 100.0 / COUNT(*),
        2
    ) AS recovery_rate
FROM patients p
JOIN treatments t ON p.patient_id = t.patient_id
JOIN outcomes o ON p.patient_id = o.patient_id
GROUP BY treatment_type
ORDER BY recovery_rate DESC;

-- ============================================
-- 4. Age Group Analysis
-- ============================================
SELECT
    CASE
        WHEN age < 18 THEN 'Under 18'
        WHEN age BETWEEN 18 AND 35 THEN '18-35'
        WHEN age BETWEEN 36 AND 50 THEN '36-50'
        WHEN age BETWEEN 51 AND 65 THEN '51-65'
        ELSE 'Over 65'
    END AS age_group,
    treatment_type,
    COUNT(*) AS total_patients,
    ROUND(AVG(recovery_days), 1) AS avg_recovery_days,
    ROUND(
        COUNT(CASE WHEN recovery_status = 'RECOVERED' THEN 1 END) * 100.0 / COUNT(*),
        2
    ) AS recovery_rate
FROM patients p
JOIN treatments t ON p.patient_id = t.patient_id
JOIN outcomes o ON p.patient_id = o.patient_id
GROUP BY age_group, treatment_type
ORDER BY age_group, recovery_rate DESC;

-- ============================================
-- 5. Comorbidity Impact Analysis
-- ============================================
SELECT
    comorbidities,
    treatment_type,
    COUNT(*) AS total_patients,
    ROUND(AVG(recovery_days), 1) AS avg_recovery_days,
    ROUND(AVG(hospital_stay_days), 1) AS avg_hospital_stay,
    ROUND(
        COUNT(CASE WHEN recovery_status = 'RECOVERED' THEN 1 END) * 100.0 / COUNT(*),
        2
    ) AS recovery_rate
FROM patients p
JOIN treatments t ON p.patient_id = t.patient_id
JOIN outcomes o ON p.patient_id = o.patient_id
GROUP BY comorbidities, treatment_type
ORDER BY recovery_rate DESC;

-- ============================================
-- 6. Monthly Treatment Trends
-- ============================================
SELECT
    DATE_TRUNC('month', treatment_start_date) AS month,
    treatment_type,
    COUNT(*) AS patients_treated,
    ROUND(AVG(recovery_days), 1) AS avg_recovery_days,
    ROUND(
        COUNT(CASE WHEN recovery_status = 'RECOVERED' THEN 1 END) * 100.0 / COUNT(*),
        2
    ) AS recovery_rate
FROM patients p
JOIN treatments t ON p.patient_id = t.patient_id
JOIN outcomes o ON p.patient_id = o.patient_id
GROUP BY DATE_TRUNC('month', treatment_start_date), treatment_type
ORDER BY month, treatment_type;

-- ============================================
-- 7. Feature Engineering for ML Models
-- ============================================
SELECT
    p.patient_id,
    p.age,
    CASE WHEN p.gender = 'M' THEN 1 ELSE 0 END AS gender_encoded,
    CASE
        WHEN p.age < 18 THEN 0
        WHEN p.age BETWEEN 18 AND 35 THEN 1
        WHEN p.age BETWEEN 36 AND 50 THEN 2
        WHEN p.age BETWEEN 51 AND 65 THEN 3
        ELSE 4
    END AS age_group_encoded,
    CASE
        WHEN t.treatment_type = 'DRUG_A' THEN 0
        WHEN t.treatment_type = 'DRUG_B' THEN 1
        WHEN t.treatment_type = 'DRUG_C' THEN 2
        ELSE 3
    END AS treatment_encoded,
    t.dosage,
    DATEDIFF('day', t.treatment_start_date, t.treatment_end_date) AS treatment_duration,
    CASE WHEN p.comorbidities IS NOT NULL THEN 1 ELSE 0 END AS has_comorbidity,
    o.recovery_days,
    o.hospital_stay_days,
    CASE WHEN o.recovery_status = 'RECOVERED' THEN 1 ELSE 0 END AS outcome_label
FROM patients p
JOIN treatments t ON p.patient_id = t.patient_id
JOIN outcomes o ON p.patient_id = o.patient_id
WHERE p.age IS NOT NULL
AND t.treatment_type IS NOT NULL
AND o.recovery_status IS NOT NULL
ORDER BY p.patient_id;

-- ============================================
-- 8. Treatment Effectiveness Summary Report
-- ============================================
WITH treatment_stats AS (
    SELECT
        treatment_type,
        COUNT(*) AS total_patients,
        ROUND(AVG(recovery_days), 1) AS avg_recovery_days,
        ROUND(AVG(hospital_stay_days), 1) AS avg_hospital_stay,
        ROUND(
            COUNT(CASE WHEN recovery_status = 'RECOVERED' THEN 1 END) * 100.0 / COUNT(*),
            2
        ) AS recovery_rate,
        ROUND(AVG(dosage), 2) AS avg_dosage
    FROM patients p
    JOIN treatments t ON p.patient_id = t.patient_id
    JOIN outcomes o ON p.patient_id = o.patient_id
    GROUP BY treatment_type
)
SELECT
    treatment_type,
    total_patients,
    avg_recovery_days,
    avg_hospital_stay,
    recovery_rate,
    avg_dosage,
    RANK() OVER (ORDER BY recovery_rate DESC) AS effectiveness_rank
FROM treatment_stats
ORDER BY effectiveness_rank;
