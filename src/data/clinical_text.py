"""Convert categorical tabular rows into clinical narrative text for BERT processing."""

from typing import List

import pandas as pd

from src.config import FEATURE_COLUMNS


class ClinicalTextGenerator:
    """Generates clinical narrative paragraphs from categorical feature rows.

    Each row of 10 categorical features is converted into a clinically
    meaningful English paragraph using domain-specific medical vocabulary.
    These texts are then suitable for BioBERT / ClinicalBERT embedding.
    """

    TEMPLATES = {
        "BirthWeight": {
            "WeightTooLow": "The newborn has a critically low birth weight",
            "LowWeight": "The newborn has a low birth weight",
            "NormalWeight": "The newborn has a normal birth weight",
        },
        "FamilyHistory": {
            "AboveTwoCases": (
                "with a significant family history of cardiac conditions "
                "(more than two cases)"
            ),
            "ZeroToTwoCases": (
                "with a limited family history of cardiac conditions "
                "(zero to two cases)"
            ),
            "NoCases": "with no family history of cardiac conditions",
        },
        "PretermBirth": {
            "4orMoreWeeksEarlier": "born four or more weeks premature",
            "2To4weeksEarlier": "born two to four weeks premature",
            "NotaPreTerm": "born at full term",
        },
        "HeartRate": {
            "RapidHeartRate": (
                "presenting with a rapid heart rate indicating tachycardia"
            ),
            "HighHeartRate": "presenting with an elevated heart rate",
            "NormalHeartRate": (
                "with a normal heart rate within expected neonatal range"
            ),
        },
        "BreathingDifficulty": {
            "HighBreathingDifficulty": "exhibiting severe respiratory distress",
            "BreathingDifficulty": "exhibiting moderate breathing difficulty",
            "NoBreathingDifficulty": "with no signs of respiratory distress",
        },
        "SkinTinge": {
            "Bluish": (
                "with cyanotic skin coloration suggesting poor oxygenation"
            ),
            "LightBluish": "with mild cyanotic skin tinge",
            "NotBluish": "with normal skin color and adequate oxygenation",
        },
        "Responsiveness": {
            "UnResponsive": "The infant is unresponsive to external stimuli",
            "SemiResponsive": (
                "The infant shows limited responsiveness to stimuli"
            ),
            "Responsive": "The infant is responsive to external stimuli",
        },
        "Movement": {
            "Diminished": "with severely diminished motor activity",
            "Decreased": "with decreased motor activity",
            "NormalMovement": "with normal motor activity and movement patterns",
        },
        "DeliveryType": {
            "C_Section": "Delivery was via cesarean section",
            "DifficultDelivery": "Delivery was classified as difficult",
            "NormalDelivery": "Delivery was normal and uncomplicated",
        },
        "MothersBPHistory": {
            "VeryHighBP": (
                "The mother has a history of very high blood pressure "
                "indicating severe hypertension"
            ),
            "HighBP": "The mother has a history of high blood pressure",
            "BPInRange": (
                "The mother's blood pressure history is within normal range"
            ),
        },
    }

    def row_to_clinical_text(self, row: pd.Series) -> str:
        """Convert a single DataFrame row of raw categorical values to clinical text.

        Args:
            row: A pandas Series with the raw (string) categorical values.

        Returns:
            A 2-4 sentence clinical paragraph.
        """
        # Sentence 1: birth weight + family history + preterm status
        s1_parts = []
        for col in ["BirthWeight", "FamilyHistory", "PretermBirth"]:
            val = str(row[col]).strip()
            phrase = self.TEMPLATES[col].get(val, "")
            if phrase:
                s1_parts.append(phrase)
        sentence1 = ", ".join(s1_parts) + "."

        # Sentence 2: heart rate + breathing + skin + responsiveness + movement
        s2_parts = []
        for col in ["HeartRate", "BreathingDifficulty", "SkinTinge"]:
            val = str(row[col]).strip()
            phrase = self.TEMPLATES[col].get(val, "")
            if phrase:
                s2_parts.append(phrase)
        sentence2 = (
            "The infant is " + ", ".join(s2_parts) + "."
            if s2_parts
            else ""
        )

        # Sentence 3: responsiveness + movement
        s3_parts = []
        for col in ["Responsiveness", "Movement"]:
            val = str(row[col]).strip()
            phrase = self.TEMPLATES[col].get(val, "")
            if phrase:
                s3_parts.append(phrase)
        sentence3 = " ".join(s3_parts) + "." if s3_parts else ""

        # Sentence 4: delivery + mother's BP
        s4_parts = []
        for col in ["DeliveryType", "MothersBPHistory"]:
            val = str(row[col]).strip()
            phrase = self.TEMPLATES[col].get(val, "")
            if phrase:
                s4_parts.append(phrase)
        sentence4 = " ".join(s4_parts) + "." if s4_parts else ""

        return " ".join(
            s for s in [sentence1, sentence2, sentence3, sentence4] if s
        )

    def generate_all(self, df_raw: pd.DataFrame) -> List[str]:
        """Generate clinical text for every row in the raw DataFrame.

        Args:
            df_raw: DataFrame with original categorical string values.

        Returns:
            List of clinical text strings, one per row.
        """
        return [self.row_to_clinical_text(row) for _, row in df_raw.iterrows()]
