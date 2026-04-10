class MedicalAdvisor:
    """Simple rule-based medical advisor for identified pills.

    Rules enforced by the system:
    - Answer only about the identified pill (provided in `pill_info`).
    - Provide general medical information only; never give diagnosis
      or personalized treatment advice.
    - For condition-based questions (pregnancy, diabetes, heart,
      kidney, elderly, children) avoid yes/no and advise consulting
      a healthcare professional.
    - Always include a medical disclaimer in responses.
    """

    FALLBACK = (
        "I can provide general medicine information only.\n"
        "Please consult a healthcare professional for medical advice."
    )

    DISCLAIMER = (
        "\n\nMedical disclaimer: This is general information and not medical "
        "advice. For personal guidance, consult a qualified healthcare professional."
    )

    def __init__(self):
        pass

    def answer_question(self, pill_info: dict, question: str) -> str:
        """Return a concise answer about the identified pill according to rules.

        pill_info should include keys like: name, dosage, usage, side_effects,
        precautions, consumption_time, confidence.
        """
        if not pill_info or not isinstance(pill_info, dict):
            return (
                "Pill information is unavailable. " + self.FALLBACK + self.DISCLAIMER
            )

        q = (question or "").strip().lower()

        # Outside-scope checks: direct patient-specific decisions
        patient_decision_keywords = ["should i", "should we", "am i", "do i have", "do i need", "should you", "can i", "can you", "prescribe", "diagnos", "diagnose"]
        if any(k in q for k in patient_decision_keywords):
            return (
                "I can provide general information about this medication, but I cannot "
                "make patient-specific recommendations or decisions. " + self.FALLBACK + self.DISCLAIMER
            )

        # Dosage management: do not change or suggest changing dosage
        dosage_keywords = ["dose", "dosage", "increase", "decrease", "start", "stop", "how much", "how many"]
        if any(k in q for k in dosage_keywords):
            return (
                "I can describe the reported dosage for this pill, but I cannot "
                "recommend changing it. Please consult a healthcare professional. "
                + self.DISCLAIMER
            )

        # Condition-based questions (never answer yes/no directly)
        condition_keywords = ["pregnan", "pregnancy", "breastfeed", "breast feeding", "diabet", "heart", "cardiac", "kidney", "renal", "elder", "older", "child", "children", "pediatric"]
        if any(k in q for k in condition_keywords):
            considerations = (
                "For people with the condition you mentioned, general considerations include: "
                "checking interactions with other medications, monitoring organ function as appropriate, "
                "and using age- or condition-specific dosing where required. "
            )
            return (
                considerations
                + "This information is general — consult a healthcare professional to determine safety for an individual."
                + self.DISCLAIMER
            )

        # Answer common-topic questions using pill_info
        if "side" in q or "effect" in q or "side effect" in q:
            se = pill_info.get("side_effects")
            if se:
                if isinstance(se, (list, tuple)):
                    se_text = ", ".join(se)
                else:
                    se_text = str(se)
                return (f"Common reported side effects: {se_text}." + self.DISCLAIMER)
            return ("No specific side-effect information available." + self.DISCLAIMER)

        if "use" in q or "what is" in q or "indication" in q:
            usage = pill_info.get("usage")
            if usage:
                return (f"Typical use: {usage}." + self.DISCLAIMER)
            return ("Usage information is not available for this item." + self.DISCLAIMER)

        if "when" in q or "time" in q or "when to take" in q or "consum" in q:
            time = pill_info.get("consumption_time")
            if time:
                return (f"Suggested consumption time: {time}." + self.DISCLAIMER)
            return ("No consumption-time information available." + self.DISCLAIMER)

        if "precaut" in q or "warning" in q or "avoid" in q:
            prec = pill_info.get("precautions")
            if prec:
                return (f"Precautions: {prec}." + self.DISCLAIMER)
            return ("No specific precautions are listed for this pill." + self.DISCLAIMER)

        if "name" in q or "what pill" in q or "identify" in q:
            name = pill_info.get("name")
            conf = pill_info.get("confidence")
            name_text = name if name else "Unknown"
            conf_text = f" (confidence {conf})" if conf is not None else ""
            return (f"Identified pill: {name_text}{conf_text}." + self.DISCLAIMER)

        # If question is short and generic, attempt a short summary
        if len(q) < 120:
            summary_parts = []
            if pill_info.get("name"):
                summary_parts.append(f"Name: {pill_info.get('name')}")
            if pill_info.get("dosage"):
                summary_parts.append(f"Dosage: {pill_info.get('dosage')}")
            if pill_info.get("usage"):
                summary_parts.append(f"Use: {pill_info.get('usage')}")
            if summary_parts:
                return ("; ".join(summary_parts) + "." + self.DISCLAIMER)

        # Fallback for anything else
        return self.FALLBACK + self.DISCLAIMER
