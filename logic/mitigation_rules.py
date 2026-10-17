from typing import List, Dict

# ==============================================================================
# DISRUPTION THRESHOLD CONFIGURATION
# ==============================================================================
# These thresholds map the LSTM's probability [0.0 - 1.0] to physical actions.
# In a real control room, these are strictly calibrated to the specific tokamak.

WARNING_THRESHOLD = 0.40    # Early precursor (e.g., Neoclassical Tearing Mode)
CRITICAL_THRESHOLD = 0.75   # Instability growing, approaching no-return
ABORT_THRESHOLD = 0.90      # Disruption imminent, thermal quench guaranteed

def evaluate_mitigation(risk_score: float) -> List[Dict[str, str]]:
    """
    Evaluates the real-time AI risk score and returns a list of immediate 
    actuator commands to stabilize the plasma or safely abort the discharge.
    
    Args:
        risk_score (float): The disruption probability from the LSTM (0.0 to 1.0)
        
    Returns:
        List[Dict]: A list of actionable commands with system, action, and severity.
    """
    actions = []

    # 1. NOMINAL STATE
    if risk_score < WARNING_THRESHOLD:
        return [{"system": "ALL", "action": "MAINTAIN_STATE", "severity": "NOMINAL"}]

    # 2. EARLY WARNING (e.g., Tearing mode detected)
    if WARNING_THRESHOLD <= risk_score < CRITICAL_THRESHOLD:
        actions.append({
            "system": "ECH_GYROTRONS",
            "action": "STEER_TO_Q2_SURFACE",
            "severity": "WARNING",
            "description": "Align Electron Cyclotron Heating to suppress magnetic island growth."
        })
        
    # 3. CRITICAL INSTABILITY (Rapid loss of confinement likely)
    if CRITICAL_THRESHOLD <= risk_score < ABORT_THRESHOLD:
        actions.append({
            "system": "POLOIDAL_FIELD_COILS",
            "action": "INITIATE_FAST_RAMPDOWN",
            "severity": "CRITICAL",
            "description": "Begin controlled reduction of plasma current (Ip)."
        })
        # If things are critical, we keep the ECH heating active to try and save it
        actions.append({
            "system": "ECH_GYROTRONS",
            "action": "MAXIMIZE_POWER",
            "severity": "CRITICAL",
            "description": "Inject maximum localized heat to prevent thermal collapse."
        })

    # 4. IMMINENT DISRUPTION (Thermal quench is unavoidable)
    if risk_score >= ABORT_THRESHOLD:
        actions.append({
            "system": "MGI_VALVES",
            "action": "TRIGGER_MASSIVE_GAS_INJECTION",
            "severity": "ABORT",
            "description": "Fire high-pressure Neon/Argon to radiate energy and prevent vessel melting."
        })
        actions.append({
            "system": "SPI_SYSTEM",
            "action": "FIRE_SHATTERED_PELLET",
            "severity": "ABORT",
            "description": "Inject shattered cryogenic deuterium to suppress runaway electrons."
        })

    return actions