def calculate_severity(prediction_class, confidence):
    """
    Heuristic-based severity calculation.
    """
    if prediction_class.lower() == 'normal':
        return 'N/A'
    
    if confidence > 0.85:
        return 'Severe'
    elif confidence > 0.65:
        return 'Moderate'
    else:
        return 'Mild'

def get_clinical_notes(prediction_class, severity):
    """
    Returns clinical notes based on diagnosis and severity.
    """
    notes = {
        'Pneumonia': {
            'Severe': "Immediate hospitalization recommended. Broad-spectrum antibiotics and oxygen support may be required.",
            'Moderate': "Oral antibiotics prescribed. Monitor respiratory rate and oxygen levels closely.",
            'Mild': "Rest, hydration, and monitoring. Follow up in 48 hours."
        },
        'COVID-19': {
            'Severe': "Isolation and immediate medical attention. Possible steroid therapy and respiratory support.",
            'Moderate': "Strict isolation. Monitor oxygen saturation. Antiviral treatment may be considered.",
            'Mild': "Symptomatic treatment and home isolation for 7 days."
        },
        'Tuberculosis': {
            'Severe': "Start intensive phase of anti-TB treatment (DOTS). Respiratory isolation if sputum positive.",
            'Moderate': "Initiate standard anti-TB drug regimen. Monitor liver function tests.",
            'Mild': "Evaluation for latent TB or early-stage active TB. Specialized consultation required."
        },
        'Malaria': {
            'Severe': "IV antimalarial treatment required. Monitor for complications like cerebral malaria.",
            'Moderate': "Standard antimalarial regimen. Monitor parasite count.",
            'Mild': "Oral antimalarial drugs. Ensure full course completion."
        },
        'Leukemia': {
            'Severe': "Urgent oncology referral. Bone marrow biopsy recommended immediately.",
            'Moderate': "Detailed hematological workup and specialist consultation.",
            'Mild': "Follow-up blood tests and hematology consultation."
        },
        'Anemia': {
            'Severe': "Consider blood transfusion. Investigate for acute blood loss or severe nutritional deficiency.",
            'Moderate': "Iron/Vitamin B12 supplementation and dietary changes. Investigate underlying cause.",
            'Mild': "Dietary advice and oral supplements. Monitor hemoglobin levels."
        },
        'Sickle Cell': {
            'Severe': "Management of pain crisis. Hydroxyurea therapy and possible transfusion.",
            'Moderate': "Pain management and hydration. Regular monitoring for complications.",
            'Mild': "Genetic counseling and prophylactic care to prevent crises."
        },
        'Normal': {
            'N/A': "No significant abnormalities detected. Maintain regular health checkups."
        }
    }
    
    return notes.get(prediction_class, {}).get(severity, "Clinical correlation required.")
