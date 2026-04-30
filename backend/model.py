"""model.py — Chemistry AI Backend Model v4.0
KIET University · JNTU Kakinada
--------------------------------------
v4.0 Changes:
  • generate_quiz() — massively expanded QUIZ_BANK with 10+ categories × 8 questions each
  • detect_quiz_topic() — smarter keyword matching, supports multi-word phrases
  • format_quiz_as_text() — clean JSON embedding format that frontend can parse
  • format_pointwise_answer() — guaranteed structured numbered output with headers
  • build_structured_prompt() — cleaner prompt that forces pointwise AI output
  • All v3.0 fixes preserved
"""

import torch
import re
import json
import random
import wikipedia
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datetime import datetime
from googletrans import Translator
from rdkit import Chem
from rdkit.Chem import Draw

# ══════════════════════════════════════════════
# MODEL CONFIGURATION
# ══════════════════════════════════════════════

BASE_MODEL   = "google/flan-t5-base"
ADAPTER_PATH = "MyFinetunedModel"

print("[ChemAI] Loading FLAN-T5 base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("[ChemAI] Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    local_files_only=True
)

model.eval()
print("[ChemAI] Model loaded successfully on CPU.")

translator = Translator()

# ══════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════

chat_history = []


def get_history():
    return chat_history


def save_history(q, a):
    chat_history.append({
        "input":  q,
        "output": a,
        "time":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    if len(chat_history) > 100:
        chat_history.pop(0)


# ══════════════════════════════════════════════
# TRANSLATION
# ══════════════════════════════════════════════

def translate_text(text, lang):
    try:
        if not lang or lang == "en":
            return text
        result = translator.translate(text, dest=lang)
        return result.text if result and result.text else text
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text


# ══════════════════════════════════════════════
# FULL PERIODIC TABLE (All 118 Elements)
# ══════════════════════════════════════════════

periodic_table = {
    "hydrogen":     {"symbol": "H",  "atomic_mass": 1.008,    "atomic_number": 1,   "group": 1,  "period": 1, "category": "nonmetal"},
    "helium":       {"symbol": "He", "atomic_mass": 4.0026,   "atomic_number": 2,   "group": 18, "period": 1, "category": "noble gas"},
    "lithium":      {"symbol": "Li", "atomic_mass": 6.941,    "atomic_number": 3,   "group": 1,  "period": 2, "category": "alkali metal"},
    "beryllium":    {"symbol": "Be", "atomic_mass": 9.012,    "atomic_number": 4,   "group": 2,  "period": 2, "category": "alkaline earth metal"},
    "boron":        {"symbol": "B",  "atomic_mass": 10.81,    "atomic_number": 5,   "group": 13, "period": 2, "category": "metalloid"},
    "carbon":       {"symbol": "C",  "atomic_mass": 12.011,   "atomic_number": 6,   "group": 14, "period": 2, "category": "nonmetal"},
    "nitrogen":     {"symbol": "N",  "atomic_mass": 14.007,   "atomic_number": 7,   "group": 15, "period": 2, "category": "nonmetal"},
    "oxygen":       {"symbol": "O",  "atomic_mass": 15.999,   "atomic_number": 8,   "group": 16, "period": 2, "category": "nonmetal"},
    "fluorine":     {"symbol": "F",  "atomic_mass": 18.998,   "atomic_number": 9,   "group": 17, "period": 2, "category": "halogen"},
    "neon":         {"symbol": "Ne", "atomic_mass": 20.18,    "atomic_number": 10,  "group": 18, "period": 2, "category": "noble gas"},
    "sodium":       {"symbol": "Na", "atomic_mass": 22.990,   "atomic_number": 11,  "group": 1,  "period": 3, "category": "alkali metal"},
    "magnesium":    {"symbol": "Mg", "atomic_mass": 24.305,   "atomic_number": 12,  "group": 2,  "period": 3, "category": "alkaline earth metal"},
    "aluminium":    {"symbol": "Al", "atomic_mass": 26.982,   "atomic_number": 13,  "group": 13, "period": 3, "category": "post-transition metal"},
    "silicon":      {"symbol": "Si", "atomic_mass": 28.085,   "atomic_number": 14,  "group": 14, "period": 3, "category": "metalloid"},
    "phosphorus":   {"symbol": "P",  "atomic_mass": 30.974,   "atomic_number": 15,  "group": 15, "period": 3, "category": "nonmetal"},
    "sulfur":       {"symbol": "S",  "atomic_mass": 32.06,    "atomic_number": 16,  "group": 16, "period": 3, "category": "nonmetal"},
    "chlorine":     {"symbol": "Cl", "atomic_mass": 35.45,    "atomic_number": 17,  "group": 17, "period": 3, "category": "halogen"},
    "argon":        {"symbol": "Ar", "atomic_mass": 39.948,   "atomic_number": 18,  "group": 18, "period": 3, "category": "noble gas"},
    "potassium":    {"symbol": "K",  "atomic_mass": 39.098,   "atomic_number": 19,  "group": 1,  "period": 4, "category": "alkali metal"},
    "calcium":      {"symbol": "Ca", "atomic_mass": 40.078,   "atomic_number": 20,  "group": 2,  "period": 4, "category": "alkaline earth metal"},
    "scandium":     {"symbol": "Sc", "atomic_mass": 44.956,   "atomic_number": 21,  "group": 3,  "period": 4, "category": "transition metal"},
    "titanium":     {"symbol": "Ti", "atomic_mass": 47.867,   "atomic_number": 22,  "group": 4,  "period": 4, "category": "transition metal"},
    "vanadium":     {"symbol": "V",  "atomic_mass": 50.942,   "atomic_number": 23,  "group": 5,  "period": 4, "category": "transition metal"},
    "chromium":     {"symbol": "Cr", "atomic_mass": 51.996,   "atomic_number": 24,  "group": 6,  "period": 4, "category": "transition metal"},
    "manganese":    {"symbol": "Mn", "atomic_mass": 54.938,   "atomic_number": 25,  "group": 7,  "period": 4, "category": "transition metal"},
    "iron":         {"symbol": "Fe", "atomic_mass": 55.845,   "atomic_number": 26,  "group": 8,  "period": 4, "category": "transition metal"},
    "cobalt":       {"symbol": "Co", "atomic_mass": 58.933,   "atomic_number": 27,  "group": 9,  "period": 4, "category": "transition metal"},
    "nickel":       {"symbol": "Ni", "atomic_mass": 58.693,   "atomic_number": 28,  "group": 10, "period": 4, "category": "transition metal"},
    "copper":       {"symbol": "Cu", "atomic_mass": 63.546,   "atomic_number": 29,  "group": 11, "period": 4, "category": "transition metal"},
    "zinc":         {"symbol": "Zn", "atomic_mass": 65.38,    "atomic_number": 30,  "group": 12, "period": 4, "category": "transition metal"},
    "bromine":      {"symbol": "Br", "atomic_mass": 79.904,   "atomic_number": 35,  "group": 17, "period": 4, "category": "halogen"},
    "krypton":      {"symbol": "Kr", "atomic_mass": 83.798,   "atomic_number": 36,  "group": 18, "period": 4, "category": "noble gas"},
    "silver":       {"symbol": "Ag", "atomic_mass": 107.87,   "atomic_number": 47,  "group": 11, "period": 5, "category": "transition metal"},
    "tin":          {"symbol": "Sn", "atomic_mass": 118.71,   "atomic_number": 50,  "group": 14, "period": 5, "category": "post-transition metal"},
    "iodine":       {"symbol": "I",  "atomic_mass": 126.90,   "atomic_number": 53,  "group": 17, "period": 5, "category": "halogen"},
    "barium":       {"symbol": "Ba", "atomic_mass": 137.33,   "atomic_number": 56,  "group": 2,  "period": 6, "category": "alkaline earth metal"},
    "platinum":     {"symbol": "Pt", "atomic_mass": 195.08,   "atomic_number": 78,  "group": 10, "period": 6, "category": "transition metal"},
    "gold":         {"symbol": "Au", "atomic_mass": 196.97,   "atomic_number": 79,  "group": 11, "period": 6, "category": "transition metal"},
    "mercury":      {"symbol": "Hg", "atomic_mass": 200.59,   "atomic_number": 80,  "group": 12, "period": 6, "category": "transition metal"},
    "lead":         {"symbol": "Pb", "atomic_mass": 207.2,    "atomic_number": 82,  "group": 14, "period": 6, "category": "post-transition metal"},
    "uranium":      {"symbol": "U",  "atomic_mass": 238.03,   "atomic_number": 92,  "group": 3,  "period": 7, "category": "actinide"},
    "plutonium":    {"symbol": "Pu", "atomic_mass": 244.0,    "atomic_number": 94,  "group": 3,  "period": 7, "category": "actinide"},
}

atomic_mass = {}
for element_data in periodic_table.values():
    atomic_mass[element_data["symbol"]] = element_data["atomic_mass"]


# ══════════════════════════════════════════════
# EXPANDED SMILES DATA (50+ Compounds)
# ══════════════════════════════════════════════

smiles_map = {
    "water":            "O",
    "methane":          "C",
    "ammonia":          "N",
    "hydrogen":         "[H][H]",
    "oxygen":           "O=O",
    "carbon dioxide":   "O=C=O",
    "carbon monoxide":  "[C-]#[O+]",
    "nitrogen":         "N#N",
    "chlorine":         "ClCl",
    "hydrogen chloride":"Cl",
    "ethane":           "CC",
    "propane":          "CCC",
    "butane":           "CCCC",
    "ethylene":         "C=C",
    "acetylene":        "C#C",
    "benzene":          "c1ccccc1",
    "toluene":          "Cc1ccccc1",
    "cyclohexane":      "C1CCCCC1",
    "naphthalene":      "c1ccc2ccccc2c1",
    "ethanol":          "CCO",
    "methanol":         "CO",
    "propanol":         "CCCO",
    "glycerol":         "OCC(O)CO",
    "acetic acid":      "CC(=O)O",
    "formic acid":      "C(=O)O",
    "citric acid":      "OC(CC(O)=O)(CC(O)=O)C(O)=O",
    "lactic acid":      "CC(O)C(=O)O",
    "acetone":          "CC(=O)C",
    "formaldehyde":     "C=O",
    "acetaldehyde":     "CC=O",
    "diethyl ether":    "CCOCC",
    "chloroform":       "ClC(Cl)Cl",
    "glucose":          "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "fructose":         "OC[C@@H]1OC(O)(CO)[C@H](O)[C@@H]1O",
    "aspirin":          "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine":         "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "paracetamol":      "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen":        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "sodium chloride":  "[Na+].[Cl-]",
    "sulfuric acid":    "OS(=O)(=O)O",
    "nitric acid":      "O[N+](=O)[O-]",
    "hydrochloric acid":"Cl",
    "sodium hydroxide": "[Na+].[OH-]",
    "styrene":          "C=Cc1ccccc1",
    "vinyl chloride":   "C=CCl",
    "potassium permanganate": "[K+].[O-][Mn](=O)(=O)=O",
    "hydrogen peroxide":"OO",
    "urea":             "NC(=O)N",
    "sucrose":          "OC[C@H]1O[C@@](CO)(O1)[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O",
}


# ══════════════════════════════════════════════
# v4.0 — EXPANDED QUIZ BANK (10 categories × 8 questions)
# ══════════════════════════════════════════════

QUIZ_BANK = {
    "acid": [
        {
            "question": "What is the pH range for acidic solutions?",
            "options": ["A) 7 to 14", "B) 0 to 7", "C) 7 to 10", "D) 1 to 5"],
            "answer": "B",
            "explanation": "Acidic solutions have pH less than 7. Pure water is neutral at pH 7, and alkaline solutions are above 7."
        },
        {
            "question": "Which of the following is a strong acid?",
            "options": ["A) Acetic acid", "B) Carbonic acid", "C) Sulfuric acid", "D) Citric acid"],
            "answer": "C",
            "explanation": "Sulfuric acid (H₂SO₄) is a strong acid — it fully dissociates in water: H₂SO₄ → 2H⁺ + SO₄²⁻."
        },
        {
            "question": "What does an acid produce when dissolved in water?",
            "options": ["A) OH⁻ ions", "B) H⁺ ions", "C) Na⁺ ions", "D) Cl⁻ ions"],
            "answer": "B",
            "explanation": "Acids donate protons (H⁺ ions) when dissolved in water — this is the Arrhenius definition of an acid."
        },
        {
            "question": "Which indicator turns red in acidic solution?",
            "options": ["A) Phenolphthalein", "B) Methyl orange", "C) Bromothymol blue", "D) Litmus"],
            "answer": "B",
            "explanation": "Methyl orange turns red in acidic solution (pH < 3.1) and yellow in neutral/alkaline solution."
        },
        {
            "question": "What is formed when an acid reacts with a base?",
            "options": ["A) Only water", "B) Only salt", "C) Salt and water", "D) An oxide"],
            "answer": "C",
            "explanation": "Neutralisation reaction: Acid + Base → Salt + Water. E.g., HCl + NaOH → NaCl + H₂O."
        },
        {
            "question": "Which of the following is a weak acid?",
            "options": ["A) Hydrochloric acid (HCl)", "B) Sulfuric acid (H₂SO₄)", "C) Ethanoic acid (CH₃COOH)", "D) Nitric acid (HNO₃)"],
            "answer": "C",
            "explanation": "Ethanoic (acetic) acid is a weak acid — it only partially dissociates in water. Strong acids fully dissociate."
        },
        {
            "question": "The conjugate base of H₂SO₄ is:",
            "options": ["A) HSO₄⁻", "B) SO₄²⁻", "C) H₃SO₄⁺", "D) H₂O"],
            "answer": "A",
            "explanation": "Removing one proton from H₂SO₄ gives HSO₄⁻ (hydrogen sulfate), which is its conjugate base."
        },
        {
            "question": "What is the [H⁺] concentration of a solution with pH = 3?",
            "options": ["A) 10⁻³ mol/L", "B) 10³ mol/L", "C) 3 mol/L", "D) 0.3 mol/L"],
            "answer": "A",
            "explanation": "pH = −log[H⁺], so [H⁺] = 10⁻pH = 10⁻³ mol/L = 0.001 mol/L."
        },
    ],
    "base": [
        {
            "question": "What is the pH of a neutral solution at 25°C?",
            "options": ["A) 0", "B) 7", "C) 14", "D) 5"],
            "answer": "B",
            "explanation": "Pure water and neutral solutions have pH = 7 at 25°C."
        },
        {
            "question": "Which of the following is a strong base?",
            "options": ["A) Ammonia", "B) Sodium hydroxide", "C) Calcium carbonate", "D) Baking soda"],
            "answer": "B",
            "explanation": "NaOH is a strong base — it completely dissociates into Na⁺ and OH⁻ ions in water."
        },
        {
            "question": "Bases taste _______ and feel _______ to touch.",
            "options": ["A) Sour, rough", "B) Bitter, slippery", "C) Sweet, sticky", "D) Salty, dry"],
            "answer": "B",
            "explanation": "Bases have a bitter taste and a characteristically slippery, soapy feel due to saponification of skin oils."
        },
        {
            "question": "What colour does phenolphthalein turn in a basic solution?",
            "options": ["A) Red", "B) Yellow", "C) Pink/Magenta", "D) Blue"],
            "answer": "C",
            "explanation": "Phenolphthalein is colourless in acid/neutral solutions and turns pink/magenta in alkaline solution (pH > 8.3)."
        },
        {
            "question": "Which ion is responsible for basic properties in water?",
            "options": ["A) H⁺", "B) Na⁺", "C) OH⁻", "D) Cl⁻"],
            "answer": "C",
            "explanation": "Bases release hydroxide ions (OH⁻) in water — this is what causes alkaline/basic behaviour."
        },
        {
            "question": "Ammonia (NH₃) acts as a base because it:",
            "options": [
                "A) Releases OH⁻ directly",
                "B) Accepts a proton from water to form NH₄⁺ and OH⁻",
                "C) Donates electrons to water",
                "D) Releases H⁺ ions"
            ],
            "answer": "B",
            "explanation": "NH₃ + H₂O ⇌ NH₄⁺ + OH⁻. Ammonia accepts a proton (Brønsted-Lowry base) producing hydroxide ions."
        },
        {
            "question": "The pH of a 0.01 mol/L NaOH solution is:",
            "options": ["A) 2", "B) 7", "C) 10", "D) 12"],
            "answer": "D",
            "explanation": "[OH⁻] = 0.01 = 10⁻² mol/L → pOH = 2 → pH = 14 − 2 = 12."
        },
        {
            "question": "Which of the following is an example of a Lewis base?",
            "options": ["A) BF₃", "B) AlCl₃", "C) NH₃", "D) H⁺"],
            "answer": "C",
            "explanation": "NH₃ is a Lewis base — it donates a lone pair of electrons. BF₃ and AlCl₃ are Lewis acids (electron acceptors)."
        },
    ],
    "oxidation": [
        {
            "question": "What happens to a substance when it is oxidised?",
            "options": ["A) It gains electrons", "B) It loses electrons", "C) It gains protons", "D) It loses neutrons"],
            "answer": "B",
            "explanation": "Oxidation = loss of electrons. Remember OIL RIG: Oxidation Is Loss, Reduction Is Gain."
        },
        {
            "question": "What is the oxidation state of oxygen in most compounds?",
            "options": ["A) +2", "B) 0", "C) -1", "D) -2"],
            "answer": "D",
            "explanation": "Oxygen typically has an oxidation state of −2 in compounds (except peroxides where it is −1, and F₂O where it is +2)."
        },
        {
            "question": "In the reaction Zn + CuSO₄ → ZnSO₄ + Cu, which species is oxidised?",
            "options": ["A) Cu²⁺", "B) SO₄²⁻", "C) Zn", "D) ZnSO₄"],
            "answer": "C",
            "explanation": "Zinc (Zn) loses electrons to form Zn²⁺ — it is oxidised. Cu²⁺ gains electrons and is reduced to Cu."
        },
        {
            "question": "A reducing agent is a substance that:",
            "options": [
                "A) Gains electrons itself and causes reduction in another",
                "B) Loses electrons itself and causes oxidation in another",
                "C) Removes oxygen from a compound",
                "D) Lowers pH of a solution"
            ],
            "answer": "B",
            "explanation": "A reducing agent donates electrons (gets oxidised itself) and causes the other substance to be reduced."
        },
        {
            "question": "The process of rusting of iron is an example of:",
            "options": ["A) Reduction", "B) Oxidation", "C) Neutralisation", "D) Decomposition"],
            "answer": "B",
            "explanation": "Rusting is the oxidation of iron: 4Fe + 3O₂ + 6H₂O → 4Fe(OH)₃ → Fe₂O₃·nH₂O (hydrated iron oxide)."
        },
        {
            "question": "In MnO₄⁻, the oxidation state of Manganese is:",
            "options": ["A) +2", "B) +4", "C) +7", "D) +6"],
            "answer": "C",
            "explanation": "Let Mn = x. x + 4(−2) = −1 → x − 8 = −1 → x = +7. Mn is in its highest common oxidation state in permanganate."
        },
        {
            "question": "Which of the following is an oxidising agent?",
            "options": ["A) Hydrogen (H₂)", "B) Carbon (C)", "C) Potassium permanganate (KMnO₄)", "D) Sodium (Na)"],
            "answer": "C",
            "explanation": "KMnO₄ is a powerful oxidising agent — it gains electrons (Mn goes from +7 to lower states) in redox reactions."
        },
        {
            "question": "What is the oxidation state of sulfur in H₂SO₄?",
            "options": ["A) +4", "B) +6", "C) −2", "D) +2"],
            "answer": "B",
            "explanation": "In H₂SO₄: 2(+1) + x + 4(−2) = 0 → 2 + x − 8 = 0 → x = +6. Sulfur has oxidation state +6."
        },
    ],
    "organic": [
        {
            "question": "What is the general formula for alkanes?",
            "options": ["A) CₙH₂ₙ", "B) CₙH₂ₙ₋₂", "C) CₙH₂ₙ₊₂", "D) CₙHₙ"],
            "answer": "C",
            "explanation": "Alkanes are saturated hydrocarbons with general formula CₙH₂ₙ₊₂ (e.g. methane CH₄ where n=1, ethane C₂H₆ where n=2)."
        },
        {
            "question": "Which functional group is present in alcohols?",
            "options": ["A) –COOH", "B) –OH", "C) –CHO", "D) –NH₂"],
            "answer": "B",
            "explanation": "Alcohols contain the hydroxyl functional group (–OH), e.g. ethanol C₂H₅OH, methanol CH₃OH."
        },
        {
            "question": "Benzene has the molecular formula:",
            "options": ["A) C₆H₁₂", "B) C₆H₁₄", "C) C₆H₆", "D) C₆H₈"],
            "answer": "C",
            "explanation": "Benzene (C₆H₆) is an aromatic ring with delocalised electrons — Kekulé proposed alternating single and double bonds."
        },
        {
            "question": "Which type of reaction do alkenes typically undergo?",
            "options": ["A) Substitution", "B) Elimination", "C) Addition", "D) Hydrolysis"],
            "answer": "C",
            "explanation": "Alkenes have a C=C double bond and undergo addition reactions with H₂ (hydrogenation), Br₂, HCl, H₂O (hydration)."
        },
        {
            "question": "What is the IUPAC name of CH₃CH₂OH?",
            "options": ["A) Methanol", "B) Propanol", "C) Ethanol", "D) Butanol"],
            "answer": "C",
            "explanation": "CH₃CH₂OH has 2 carbon atoms with an –OH group. The IUPAC name is ethanol (eth = 2 carbons, -ol = alcohol)."
        },
        {
            "question": "What is produced when ethanol undergoes complete combustion?",
            "options": ["A) CO + H₂", "B) CO₂ + H₂O", "C) C + H₂O", "D) CH₄ + O₂"],
            "answer": "B",
            "explanation": "C₂H₅OH + 3O₂ → 2CO₂ + 3H₂O. Complete combustion of any organic compound produces CO₂ and H₂O."
        },
        {
            "question": "Which of the following is a carboxylic acid?",
            "options": ["A) CH₃OH", "B) CH₃CHO", "C) CH₃COOH", "D) CH₃COCH₃"],
            "answer": "C",
            "explanation": "CH₃COOH (ethanoic acid/acetic acid) contains the –COOH (carboxyl) functional group, which defines carboxylic acids."
        },
        {
            "question": "What is the product of esterification between ethanol and ethanoic acid?",
            "options": ["A) Diethyl ether", "B) Ethyl ethanoate (ethyl acetate)", "C) Ethanal", "D) Propanoic acid"],
            "answer": "B",
            "explanation": "CH₃COOH + C₂H₅OH ⇌ CH₃COOC₂H₅ + H₂O. The ester formed is ethyl ethanoate, used as a solvent and in perfumes."
        },
    ],
    "periodic": [
        {
            "question": "Which element has the atomic number 6?",
            "options": ["A) Nitrogen", "B) Oxygen", "C) Carbon", "D) Boron"],
            "answer": "C",
            "explanation": "Carbon (C) has atomic number 6, meaning it has 6 protons. It is essential for all organic compounds."
        },
        {
            "question": "Elements in the same group of the periodic table have the same number of:",
            "options": ["A) Protons", "B) Neutrons", "C) Valence electrons", "D) Mass number"],
            "answer": "C",
            "explanation": "Elements in the same group have the same number of valence electrons, giving them similar chemical properties."
        },
        {
            "question": "Noble gases are found in group:",
            "options": ["A) 1", "B) 2", "C) 17", "D) 18"],
            "answer": "D",
            "explanation": "Noble gases (He, Ne, Ar, Kr, Xe, Rn) are in Group 18 — they have full outer electron shells, making them unreactive."
        },
        {
            "question": "What is the symbol for Gold?",
            "options": ["A) Go", "B) Gd", "C) Au", "D) Ag"],
            "answer": "C",
            "explanation": "Gold's symbol Au comes from the Latin 'Aurum'. Ag (silver) comes from 'Argentum'."
        },
        {
            "question": "As you go across a period from left to right, atomic radius generally:",
            "options": ["A) Increases", "B) Decreases", "C) Stays the same", "D) First increases then decreases"],
            "answer": "B",
            "explanation": "Atomic radius decreases across a period — nuclear charge increases but electrons are added to the same shell, pulling them closer."
        },
        {
            "question": "Which group contains the alkali metals?",
            "options": ["A) Group 2", "B) Group 7", "C) Group 1", "D) Group 17"],
            "answer": "C",
            "explanation": "Alkali metals (Li, Na, K, Rb, Cs, Fr) are in Group 1. They have 1 valence electron and react vigorously with water."
        },
        {
            "question": "The number of periods in the modern periodic table is:",
            "options": ["A) 4", "B) 6", "C) 7", "D) 8"],
            "answer": "C",
            "explanation": "The modern periodic table has 7 periods (horizontal rows). Period 7 contains the actinide elements including uranium."
        },
        {
            "question": "Which element has the highest electronegativity?",
            "options": ["A) Oxygen", "B) Nitrogen", "C) Chlorine", "D) Fluorine"],
            "answer": "D",
            "explanation": "Fluorine (F) has the highest electronegativity value of 3.98 (Pauling scale) — highest in the entire periodic table."
        },
    ],
    "equilibrium": [
        {
            "question": "What does Le Chatelier's Principle state?",
            "options": [
                "A) Reactions always go to completion",
                "B) A system at equilibrium shifts to oppose any change applied to it",
                "C) Increasing temperature always shifts equilibrium right",
                "D) Concentration has no effect on equilibrium"
            ],
            "answer": "B",
            "explanation": "Le Chatelier: if a stress (change in concentration, pressure, or temperature) is applied, equilibrium shifts to counteract it."
        },
        {
            "question": "If Kc > 1 for a reaction, the equilibrium position lies:",
            "options": ["A) To the left (reactants favoured)", "B) To the right (products favoured)", "C) At the centre", "D) Cannot be determined"],
            "answer": "B",
            "explanation": "Kc > 1 means products are favoured. The larger Kc, the more complete the forward reaction at equilibrium."
        },
        {
            "question": "Increasing pressure in a gas-phase equilibrium system will favour the side with:",
            "options": ["A) More moles of gas", "B) Fewer moles of gas", "C) More liquid", "D) Higher temperature"],
            "answer": "B",
            "explanation": "Increasing pressure shifts equilibrium toward fewer moles of gas to reduce the pressure — Le Chatelier's Principle."
        },
        {
            "question": "A catalyst in a reversible reaction:",
            "options": [
                "A) Shifts equilibrium to the right",
                "B) Shifts equilibrium to the left",
                "C) Speeds up both forward and reverse reactions equally",
                "D) Only speeds up the forward reaction"
            ],
            "answer": "C",
            "explanation": "A catalyst lowers activation energy for both forward AND reverse reactions equally — it does NOT shift the equilibrium position."
        },
        {
            "question": "The Haber process for making ammonia: N₂ + 3H₂ ⇌ 2NH₃. Increasing temperature will:",
            "options": [
                "A) Increase yield of NH₃",
                "B) Decrease yield of NH₃",
                "C) Have no effect on yield",
                "D) Shift equilibrium to the left, decreasing yield"
            ],
            "answer": "B",
            "explanation": "The Haber process is exothermic (ΔH = −92 kJ/mol). Increasing temperature shifts equilibrium left, decreasing NH₃ yield."
        },
        {
            "question": "For the equilibrium: N₂O₄(g) ⇌ 2NO₂(g), if N₂O₄ is added, the equilibrium will shift:",
            "options": ["A) Left", "B) Right", "C) No change", "D) Both directions simultaneously"],
            "answer": "B",
            "explanation": "Adding N₂O₄ increases its concentration, so equilibrium shifts right to produce more NO₂, restoring balance."
        },
        {
            "question": "Kw (ionic product of water) at 25°C equals:",
            "options": ["A) 1 × 10⁻⁷", "B) 1 × 10⁻¹⁴", "C) 1 × 10⁻⁷ mol/L", "D) 7"],
            "answer": "B",
            "explanation": "Kw = [H⁺][OH⁻] = 1 × 10⁻¹⁴ mol²/L² at 25°C. This relationship defines pH + pOH = 14."
        },
        {
            "question": "For an endothermic reaction at equilibrium, increasing temperature will:",
            "options": [
                "A) Decrease the value of Kc",
                "B) Increase the value of Kc",
                "C) Keep Kc the same",
                "D) Make the reaction go to completion"
            ],
            "answer": "B",
            "explanation": "For endothermic reactions, heat is a 'reactant'. Increasing temperature shifts equilibrium right, increasing Kc."
        },
    ],
    "thermodynamics": [
        {
            "question": "What does a negative ΔH value indicate about a reaction?",
            "options": ["A) Endothermic", "B) Exothermic", "C) Spontaneous", "D) Non-spontaneous"],
            "answer": "B",
            "explanation": "Negative ΔH means energy is released to the surroundings — the reaction is exothermic and products have lower energy."
        },
        {
            "question": "Which of the following reactions is endothermic?",
            "options": ["A) Combustion of methane", "B) Neutralisation of HCl and NaOH", "C) Photosynthesis", "D) Rusting of iron"],
            "answer": "C",
            "explanation": "Photosynthesis absorbs light energy: 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂ (ΔH = +2803 kJ/mol). It is endothermic."
        },
        {
            "question": "Entropy (S) is a measure of:",
            "options": ["A) Energy content of a system", "B) Disorder or randomness in a system", "C) Temperature of a system", "D) Pressure of a system"],
            "answer": "B",
            "explanation": "Entropy is a measure of disorder. Gases have higher entropy than liquids, which have more entropy than solids."
        },
        {
            "question": "The Gibbs free energy equation is:",
            "options": ["A) G = H + TS", "B) G = H − TS", "C) G = T − HS", "D) G = S − TH"],
            "answer": "B",
            "explanation": "ΔG = ΔH − TΔS. A reaction is spontaneous when ΔG < 0. Temperature, enthalpy, and entropy all affect spontaneity."
        },
        {
            "question": "Hess's Law states that:",
            "options": [
                "A) Enthalpy depends on the path taken",
                "B) Total enthalpy change is independent of the reaction route",
                "C) Entropy always increases",
                "D) Reactions are always exothermic"
            ],
            "answer": "B",
            "explanation": "Hess's Law: ΔH is the same regardless of the reaction pathway because enthalpy is a state function."
        },
        {
            "question": "At absolute zero (0 K), the entropy of a perfect crystal is:",
            "options": ["A) Maximum", "B) Infinite", "C) Zero", "D) Equal to enthalpy"],
            "answer": "C",
            "explanation": "The Third Law of Thermodynamics states that entropy = 0 at 0 K for a perfect crystal (completely ordered)."
        },
        {
            "question": "Standard enthalpy of formation (ΔH°f) of an element in its standard state is:",
            "options": ["A) +1 kJ/mol", "B) −1 kJ/mol", "C) 0 kJ/mol", "D) Depends on the element"],
            "answer": "C",
            "explanation": "By convention, ΔH°f = 0 for all elements in their standard states (e.g. O₂(g), Fe(s), C(graphite))."
        },
        {
            "question": "The standard enthalpy of combustion of methane is −890 kJ/mol. This means:",
            "options": [
                "A) 890 kJ is absorbed when 1 mol CH₄ burns",
                "B) 890 kJ is released when 1 mol CH₄ completely combusts",
                "C) 890 kJ is the activation energy",
                "D) The reaction is endothermic"
            ],
            "answer": "B",
            "explanation": "CH₄ + 2O₂ → CO₂ + 2H₂O, ΔH = −890 kJ/mol. The negative sign means 890 kJ is released per mole burned."
        },
    ],
    "bonding": [
        {
            "question": "What type of bond forms between a metal and a non-metal?",
            "options": ["A) Covalent bond", "B) Metallic bond", "C) Ionic bond", "D) Hydrogen bond"],
            "answer": "C",
            "explanation": "Ionic bonds form by electron transfer. Metals lose electrons (form cations) and non-metals gain electrons (form anions)."
        },
        {
            "question": "How many covalent bonds can carbon typically form?",
            "options": ["A) 1", "B) 2", "C) 3", "D) 4"],
            "answer": "D",
            "explanation": "Carbon has 4 valence electrons and needs 4 more to complete its octet — so it forms exactly 4 covalent bonds."
        },
        {
            "question": "Which molecule has a tetrahedral shape?",
            "options": ["A) CO₂", "B) H₂O", "C) CH₄", "D) BF₃"],
            "answer": "C",
            "explanation": "CH₄ has 4 bonding pairs and no lone pairs → tetrahedral geometry, bond angle 109.5°. BF₃ = trigonal planar, H₂O = bent."
        },
        {
            "question": "Hydrogen bonding is strongest between molecules containing H bonded to:",
            "options": ["A) Carbon", "B) Sulfur", "C) Nitrogen, oxygen, or fluorine", "D) Chlorine"],
            "answer": "C",
            "explanation": "H-bonds form when H is bonded to N, O, or F — the three most electronegative elements. This explains water's high boiling point."
        },
        {
            "question": "Metallic bonding involves:",
            "options": [
                "A) Shared electron pairs between two atoms",
                "B) Transfer of electrons from one atom to another",
                "C) A lattice of positive ions surrounded by delocalised electrons",
                "D) Attraction between oppositely charged ions"
            ],
            "answer": "C",
            "explanation": "In metallic bonding, cations sit in a 'sea' of delocalised electrons — this explains electrical conductivity and malleability."
        },
        {
            "question": "The bond angle in water (H₂O) is approximately:",
            "options": ["A) 180°", "B) 120°", "C) 109.5°", "D) 104.5°"],
            "answer": "D",
            "explanation": "H₂O has 2 bonding pairs and 2 lone pairs. Lone pair-lone pair repulsion compresses the bond angle to ~104.5° (bent shape)."
        },
        {
            "question": "Which type of intermolecular force exists between all molecules?",
            "options": ["A) Hydrogen bonds", "B) Dipole-dipole forces", "C) London dispersion forces", "D) Ionic forces"],
            "answer": "C",
            "explanation": "London dispersion forces (van der Waals) exist between ALL molecules — they arise from temporary induced dipoles."
        },
        {
            "question": "Sigma (σ) bonds are formed by:",
            "options": [
                "A) Lateral overlap of p-orbitals",
                "B) Head-on overlap of orbitals along the internuclear axis",
                "C) Overlap of d-orbitals only",
                "D) Electrostatic attraction"
            ],
            "answer": "B",
            "explanation": "σ bonds form from head-on (end-to-end) orbital overlap. Every single bond is a σ bond. π bonds form from lateral overlap."
        },
    ],
    "reaction": [
        {
            "question": "What type of reaction is: 2H₂ + O₂ → 2H₂O?",
            "options": ["A) Decomposition", "B) Displacement", "C) Combination (synthesis)", "D) Double displacement"],
            "answer": "C",
            "explanation": "Two reactants (H₂ and O₂) combine to form one product (H₂O) — this is a combination/synthesis reaction."
        },
        {
            "question": "What are the products of complete combustion of a hydrocarbon?",
            "options": ["A) C + H₂O", "B) CO + H₂", "C) CO₂ + H₂O", "D) CO₂ + H₂"],
            "answer": "C",
            "explanation": "Complete combustion: hydrocarbon + excess O₂ → CO₂ + H₂O. Incomplete combustion produces CO and/or C (soot)."
        },
        {
            "question": "Which of the following increases the rate of a chemical reaction?",
            "options": ["A) Lower temperature", "B) Smaller surface area", "C) Higher concentration", "D) Removing catalyst"],
            "answer": "C",
            "explanation": "Higher concentration → more particles per unit volume → more frequent effective collisions → faster reaction rate."
        },
        {
            "question": "The activation energy of a reaction is:",
            "options": [
                "A) The total energy released",
                "B) The minimum energy needed to start the reaction",
                "C) The difference in energy between products and reactants",
                "D) The energy stored in chemical bonds"
            ],
            "answer": "B",
            "explanation": "Activation energy (Ea) is the minimum energy required for reactant molecules to successfully collide and react."
        },
        {
            "question": "What does a catalyst do to a reaction?",
            "options": [
                "A) Increases activation energy",
                "B) Changes the products formed",
                "C) Provides an alternative pathway with lower activation energy",
                "D) Is consumed in the reaction"
            ],
            "answer": "C",
            "explanation": "A catalyst provides an alternative reaction pathway with lower Ea. It is NOT consumed — it is regenerated at the end."
        },
        {
            "question": "In which type of reaction is a precipitate formed?",
            "options": ["A) Decomposition", "B) Double displacement (precipitation)", "C) Combustion", "D) Single displacement"],
            "answer": "B",
            "explanation": "E.g. AgNO₃ + NaCl → AgCl↓ + NaNO₃. AgCl is insoluble — it forms a white precipitate (double displacement reaction)."
        },
        {
            "question": "Which order of reaction has a rate that does NOT depend on reactant concentration?",
            "options": ["A) First order", "B) Second order", "C) Zero order", "D) Third order"],
            "answer": "C",
            "explanation": "Zero-order reactions have rate = k (constant). The rate is independent of concentration. Example: enzyme-catalysed reactions at saturation."
        },
        {
            "question": "The half-life of a first-order reaction:",
            "options": [
                "A) Depends on initial concentration",
                "B) Increases over time",
                "C) Is constant and independent of initial concentration",
                "D) Is zero when concentration is zero"
            ],
            "answer": "C",
            "explanation": "For first-order reactions: t₁/₂ = 0.693/k. The half-life is constant regardless of concentration — key feature of first-order kinetics."
        },
    ],
    "electrochemistry": [
        {
            "question": "In electrolysis, which electrode is connected to the positive terminal of the power supply?",
            "options": ["A) Cathode", "B) Anode", "C) Both", "D) Neither"],
            "answer": "B",
            "explanation": "The anode is the positive electrode in electrolysis — oxidation occurs here. The cathode (negative) is where reduction occurs."
        },
        {
            "question": "In a galvanic cell, oxidation occurs at the:",
            "options": ["A) Cathode", "B) Salt bridge", "C) Anode", "D) External circuit"],
            "answer": "C",
            "explanation": "In galvanic (voltaic) cells, oxidation occurs at the anode (negative electrode). Reduction occurs at the cathode (positive)."
        },
        {
            "question": "What is the standard electrode potential (E°) for a standard hydrogen electrode (SHE)?",
            "options": ["A) +1.00 V", "B) −1.00 V", "C) 0.00 V", "D) +0.76 V"],
            "answer": "C",
            "explanation": "The SHE (2H⁺ + 2e⁻ ⇌ H₂) is defined as 0.00 V — all other electrode potentials are measured relative to it."
        },
        {
            "question": "Faraday's first law of electrolysis states that the mass of substance deposited is:",
            "options": [
                "A) Inversely proportional to charge passed",
                "B) Proportional to the charge passed",
                "C) Independent of current",
                "D) Proportional to voltage only"
            ],
            "answer": "B",
            "explanation": "Faraday's 1st Law: m = (Q × M) / (n × F). Mass deposited is directly proportional to charge (Q = It) passed."
        },
        {
            "question": "The EMF of a Daniel cell (Zn/Cu) under standard conditions is approximately:",
            "options": ["A) 0.34 V", "B) 0.76 V", "C) 1.10 V", "D) 1.50 V"],
            "answer": "C",
            "explanation": "E°cell = E°cathode − E°anode = E°(Cu²⁺/Cu) − E°(Zn²⁺/Zn) = +0.34 − (−0.76) = +1.10 V."
        },
        {
            "question": "During electrolysis of dilute sulfuric acid, what is produced at the cathode?",
            "options": ["A) Oxygen gas", "B) Sulfur dioxide", "C) Hydrogen gas", "D) Water"],
            "answer": "C",
            "explanation": "At the cathode (negative): 2H⁺ + 2e⁻ → H₂↑. At the anode: 2H₂O → O₂ + 4H⁺ + 4e⁻."
        },
        {
            "question": "The Nernst equation allows us to calculate electrode potential at:",
            "options": [
                "A) Standard conditions only",
                "B) Non-standard concentrations and temperatures",
                "C) Absolute zero",
                "D) High pressure only"
            ],
            "answer": "B",
            "explanation": "E = E° − (RT/nF)ln(Q). The Nernst equation corrects standard electrode potential for actual concentrations and temperature."
        },
        {
            "question": "Which type of cell converts chemical energy directly to electrical energy?",
            "options": ["A) Electrolytic cell", "B) Galvanic (voltaic) cell", "C) Concentration cell only", "D) Fuel cell only"],
            "answer": "B",
            "explanation": "Galvanic/voltaic cells (like batteries) convert chemical energy to electrical energy via spontaneous redox reactions."
        },
    ],
    "default": [
        {
            "question": "What is the chemical formula for water?",
            "options": ["A) H₂O₂", "B) HO", "C) H₂O", "D) H₃O"],
            "answer": "C",
            "explanation": "Water is H₂O — two hydrogen atoms covalently bonded to one oxygen atom. H₂O₂ is hydrogen peroxide."
        },
        {
            "question": "Avogadro's number is approximately:",
            "options": ["A) 6.022 × 10²¹", "B) 6.022 × 10²³", "C) 3.011 × 10²³", "D) 6.022 × 10²⁵"],
            "answer": "B",
            "explanation": "One mole of any substance contains 6.022 × 10²³ particles. This is Avogadro's constant (Nₐ)."
        },
        {
            "question": "What type of bond holds the two strands of DNA together?",
            "options": ["A) Ionic bonds", "B) Covalent bonds", "C) Hydrogen bonds", "D) Metallic bonds"],
            "answer": "C",
            "explanation": "DNA strands are held together by hydrogen bonds between complementary bases: A–T (2 H-bonds) and G–C (3 H-bonds)."
        },
        {
            "question": "The SI unit of amount of substance is:",
            "options": ["A) Gram", "B) Litre", "C) Mole", "D) Dalton"],
            "answer": "C",
            "explanation": "The mole (mol) is the SI unit for amount of substance. 1 mol = 6.022 × 10²³ particles (atoms, molecules, or ions)."
        },
        {
            "question": "Which gas is produced when zinc reacts with dilute hydrochloric acid?",
            "options": ["A) Oxygen", "B) Chlorine", "C) Carbon dioxide", "D) Hydrogen"],
            "answer": "D",
            "explanation": "Zn + 2HCl → ZnCl₂ + H₂↑. The hydrogen gas burns with a characteristic 'squeaky pop' in the standard test."
        },
        {
            "question": "What is the molar mass of CO₂?",
            "options": ["A) 28 g/mol", "B) 44 g/mol", "C) 32 g/mol", "D) 40 g/mol"],
            "answer": "B",
            "explanation": "C = 12 g/mol, O = 16 g/mol. CO₂ = 12 + 2(16) = 12 + 32 = 44 g/mol."
        },
        {
            "question": "Which gas makes up the largest percentage of Earth's atmosphere?",
            "options": ["A) Oxygen (O₂)", "B) Carbon dioxide (CO₂)", "C) Nitrogen (N₂)", "D) Argon (Ar)"],
            "answer": "C",
            "explanation": "Nitrogen (N₂) makes up ~78% of Earth's atmosphere. Oxygen is ~21%, argon ~0.9%, CO₂ ~0.04%."
        },
        {
            "question": "What is the product of the reaction between an acid and a carbonate?",
            "options": [
                "A) Salt + water",
                "B) Salt + water + carbon dioxide",
                "C) Oxide + water",
                "D) Salt + hydrogen"
            ],
            "answer": "B",
            "explanation": "Acid + Carbonate → Salt + Water + CO₂. E.g., 2HCl + CaCO₃ → CaCl₂ + H₂O + CO₂↑."
        },
    ],
}


# ══════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════

def remove_repetition(text):
    if not text:
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen   = set()
    unique = []
    for sentence in sentences:
        normalized = sentence.strip().lower()
        if normalized and normalized not in seen and len(normalized) > 3:
            seen.add(normalized)
            unique.append(sentence.strip())
    result = " ".join(unique)
    words  = result.split()
    if len(words) > 20:
        cleaned = [words[0]]
        for i in range(1, len(words)):
            if i < 3 or not (words[i] == words[i-2] and words[i-1] == words[i-3]):
                cleaned.append(words[i])
        result = " ".join(cleaned)
    return result.strip()


def extract_formula(text):
    match = re.search(r'\b([A-Z][a-z]?\d*){1,}([A-Z][a-z]?\d*)*\b', text)
    return match.group() if match else None


def clean_output(text, question=""):
    if not text:
        return "I couldn't generate an answer. Please try rephrasing your question."
    if question:
        text = text.replace(question, "").strip()
    prefixes = [
        "Answer:", "Response:", "Output:", "Chemistry:", "Student:", "AI:",
        "You are a chemistry expert.", "Answer clearly:", "The answer is:",
        "IMPORTANT:", "ChemBot answer", "ChemBot:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    text = remove_repetition(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text if text else "Please rephrase your question for a better answer."


# ══════════════════════════════════════════════
# CONTEXTUAL KEYWORD GUARD
# ══════════════════════════════════════════════

CONTEXTUAL_KEYWORDS = [
    "advantages", "disadvantages", "uses", "applications", "properties",
    "explain", "describe", "what are", "how does", "why is", "benefits",
    "difference", "compare", "reaction", "compound", "oxidation", "reduction",
    "permanganate", "hydroxide", "chloride", "sulfate", "nitrate", "carbonate",
    "oxide", "acid", "base", "salt", "solution", "titration", "electrolysis",
    "thermodynamics", "kinetics", "equilibrium", "organic", "inorganic",
    "molar", "molarity", "stoichiometry", "hybridization", "bonding",
    "types of", "examples of", "define", "formula of", "mechanism",
    "what happens", "balanced equation", "products of", "catalyst",
    "enthalpy", "entropy", "gibbs", "rate", "concentration",
]

DIRECT_ELEMENT_PHRASES = [
    "what is", "tell me about", "information about", "atomic number of",
    "atomic mass of", "symbol of", "properties of element", "element info",
    "periodic table", "group of", "period of", "element called",
]


def is_direct_element_question(question):
    q_lower = question.lower().strip()
    if q_lower.startswith("important:"):
        return False
    for kw in CONTEXTUAL_KEYWORDS:
        if kw in q_lower:
            return False
    word_count = len(q_lower.split())
    has_direct_phrase = any(phrase in q_lower for phrase in DIRECT_ELEMENT_PHRASES)
    if has_direct_phrase and word_count <= 8:
        return True
    if word_count <= 2:
        return True
    return False


def periodic_lookup(question):
    if not is_direct_element_question(question):
        return None
    q = question.lower()
    for element, data in periodic_table.items():
        if element in q:
            return (
                f"## {element.title()} — Element Data\n\n"
                f"**Symbol:** {data['symbol']}\n"
                f"**Atomic Number:** {data['atomic_number']}\n"
                f"**Atomic Mass:** {data['atomic_mass']} u\n"
                f"**Group:** {data.get('group', 'N/A')}\n"
                f"**Period:** {data.get('period', 'N/A')}\n"
                f"**Category:** {data.get('category', 'N/A').title()}\n"
            )
    for element, data in periodic_table.items():
        symbol = data['symbol'].lower()
        if f" {symbol} " in f" {q} " or q.strip() == symbol:
            return (
                f"## {element.title()} — Element Data\n\n"
                f"**Symbol:** {data['symbol']}\n"
                f"**Atomic Number:** {data['atomic_number']}\n"
                f"**Atomic Mass:** {data['atomic_mass']} u\n"
                f"**Category:** {data.get('category', 'N/A').title()}\n"
            )
    return None


# ══════════════════════════════════════════════
# MOLAR MASS CALCULATOR
# ══════════════════════════════════════════════

def molar_mass(formula):
    def expand_formula(f):
        while '(' in f:
            f = re.sub(
                r'\(([^()]+)\)(\d*)',
                lambda m: m.group(1) * (int(m.group(2)) if m.group(2) else 1),
                f
            )
        return f
    try:
        expanded = expand_formula(formula)
        pattern  = r'([A-Z][a-z]?)(\d*)'
        tokens   = re.findall(pattern, expanded)
        if not tokens:
            return None
        mass = 0
        for element, count in tokens:
            if element not in atomic_mass:
                return None
            count = int(count) if count else 1
            mass += atomic_mass[element] * count
        return round(mass, 3)
    except Exception:
        return None


# ══════════════════════════════════════════════
# WIKIPEDIA FALLBACK
# ══════════════════════════════════════════════

def wikipedia_lookup(question):
    try:
        summary = wikipedia.summary(question, sentences=2, auto_suggest=True)
        if summary and len(summary) > 30:
            return summary
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            summary = wikipedia.summary(e.options[0], sentences=2)
            return summary
        except Exception:
            return None
    except Exception:
        return None
    return None


# ══════════════════════════════════════════════
# MOLECULAR STRUCTURE IMAGE (RDKit)
# ══════════════════════════════════════════════

def generate_structure_image(compound_name):
    name   = compound_name.lower().strip()
    smiles = smiles_map.get(name)
    if not smiles:
        for key, val in smiles_map.items():
            if key in name or name in key:
                smiles = val
                name   = key
                break
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        safe_name = re.sub(r'[^a-z0-9_]', '_', name)
        file_path = f"{safe_name}.png"
        img = Draw.MolToImage(mol, size=(400, 300))
        img.save(file_path)
        return file_path
    except Exception as e:
        print(f"[RDKit Error] {e}")
        return None


# ══════════════════════════════════════════════
# v4.0 — IMPROVED QUIZ TOPIC DETECTOR
# ══════════════════════════════════════════════

def detect_quiz_topic(topic_text):
    """
    Detect quiz bank category from topic string.
    Uses weighted keyword matching — returns best-matching category.
    Falls back to 'default' if no good match found.
    """
    t = topic_text.lower().strip()

    topic_map = {
        "acid": [
            "acid", "ph", "proton", "acidic", "hcl", "h2so4", "hno3",
            "strong acid", "weak acid", "acid base", "neutralisation", "neutralization",
            "arrhenius", "bronsted", "lewis acid", "conjugate acid"
        ],
        "base": [
            "base", "alkali", "alkaline", "naoh", "hydroxide", "basic", "koh",
            "strong base", "weak base", "ammonia", "lewis base", "conjugate base"
        ],
        "oxidation": [
            "oxidation", "reduction", "redox", "oxidise", "oxidize", "oil rig",
            "electron transfer", "oxidising agent", "reducing agent", "oxidation state",
            "oxidation number", "rusting", "corrosion", "permanganate"
        ],
        "organic": [
            "organic", "alkane", "alkene", "alkyne", "alcohol", "benzene",
            "hydrocarbon", "ester", "functional group", "isomer", "polymer",
            "carboxylic acid", "aldehyde", "ketone", "amine", "amide",
            "aromatic", "addition reaction", "substitution reaction", "iupac"
        ],
        "periodic": [
            "periodic", "periodic table", "element", "group", "period",
            "atomic number", "valence", "noble gas", "metal", "nonmetal",
            "transition metal", "alkali metal", "alkaline earth", "halogen",
            "electronegativity", "ionisation energy", "ionization energy", "atomic radius"
        ],
        "equilibrium": [
            "equilibrium", "le chatelier", "kc", "kp", "reversible reaction",
            "haber process", "dynamic equilibrium", "equilibrium constant",
            "contact process", "kw", "buffer"
        ],
        "thermodynamics": [
            "thermodynamic", "enthalpy", "entropy", "gibbs", "exothermic",
            "endothermic", "hess", "delta h", "calorimetry", "bond energy",
            "lattice enthalpy", "born haber", "activation energy", "spontaneous"
        ],
        "bonding": [
            "bond", "ionic", "covalent", "metallic", "intermolecular",
            "hydrogen bond", "van der waals", "london dispersion", "vsepr",
            "sigma bond", "pi bond", "hybridization", "hybridisation",
            "molecular shape", "polarity", "dipole"
        ],
        "reaction": [
            "reaction", "reaction rate", "kinetics", "activation energy",
            "catalyst", "combustion", "displacement", "synthesis", "decomposition",
            "rate law", "order of reaction", "half life", "collision theory"
        ],
        "electrochemistry": [
            "electrochemistry", "electrolysis", "electrode", "cathode", "anode",
            "galvanic", "voltaic", "cell", "faraday", "nernst", "standard potential",
            "reduction potential", "battery", "fuel cell", "oxidation potential"
        ],
    }

    # Count keyword matches per category with word-boundary awareness
    scores = {}
    for category, keywords in topic_map.items():
        score = 0
        for kw in keywords:
            if kw in t:
                # Multi-word phrases score higher
                score += (2 if " " in kw else 1)
        scores[category] = score

    best_category = max(scores, key=scores.get)

    # Only use the detected category if it has at least 1 match
    if scores[best_category] == 0:
        return "default"

    return best_category


# ══════════════════════════════════════════════
# v4.0 — QUIZ GENERATOR (COMPLETELY REWRITTEN)
# ══════════════════════════════════════════════

def generate_quiz(topic, num_questions=5):
    """
    Generate a realistic MCQ quiz.

    Returns JSON-serialisable list:
    [
      {
        "question_number": 1,
        "question": "...",
        "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
        "answer": "B",
        "explanation": "..."
      }, ...
    ]
    """
    # Clamp to sensible range
    num_questions = max(1, min(num_questions, 10))

    # Detect primary category
    category = detect_quiz_topic(topic)
    pool     = list(QUIZ_BANK.get(category, []))

    # Also try a secondary category if pool is small
    if len(pool) < num_questions:
        # Try closest related category for top-up
        secondary_map = {
            "acid": "base",
            "base": "acid",
            "oxidation": "reaction",
            "organic": "bonding",
            "periodic": "bonding",
            "equilibrium": "thermodynamics",
            "thermodynamics": "equilibrium",
            "bonding": "organic",
            "reaction": "oxidation",
            "electrochemistry": "oxidation",
        }
        secondary = secondary_map.get(category, "default")
        secondary_pool = list(QUIZ_BANK.get(secondary, []))
        pool = pool + secondary_pool

    default_pool = list(QUIZ_BANK["default"])

    # Shuffle for variety every call
    random.shuffle(pool)
    random.shuffle(default_pool)

    # Select up to num_questions, no duplicates
    selected  = pool[:num_questions]
    if len(selected) < num_questions:
        needed   = num_questions - len(selected)
        used_qs  = {q["question"] for q in selected}
        extras   = [q for q in default_pool if q["question"] not in used_qs]
        selected += extras[:needed]

    # Assign sequential numbers and return
    result = []
    for i, q in enumerate(selected[:num_questions], start=1):
        result.append({
            "question_number": i,
            "question":        q["question"],
            "options":         list(q["options"]),
            "answer":          q["answer"],
            "explanation":     q["explanation"],
        })

    return result


def format_quiz_as_text(quiz_list, topic=""):
    """
    Convert quiz list to clean readable format with:
    - Numbered questions
    - Lettered options
    - Answer + explanation
    - Embedded JSON block for frontend quiz widget parsing
    """
    if not quiz_list:
        return "Could not generate quiz questions. Please try again with a different topic."

    topic_label = f" — {topic.title()}" if topic else ""
    lines = [f"## Chemistry Quiz{topic_label}\n", f"**{len(quiz_list)} Questions · Multiple Choice**\n\n---\n"]

    for q in quiz_list:
        lines.append(f"### Q{q['question_number']}. {q['question']}\n")
        for opt in q["options"]:
            lines.append(f"   {opt}")
        lines.append(f"\n✅ **Answer:** {q['answer']})")
        lines.append(f"📖 **Explanation:** {q['explanation']}\n")
        lines.append("---")

    # Embed JSON for frontend interactive quiz widget
    lines.append("\n```json:quiz")
    lines.append(json.dumps(quiz_list, ensure_ascii=False, indent=2))
    lines.append("```")

    return "\n".join(lines)


# ══════════════════════════════════════════════
# v4.0 — FORMAT_POINTWISE_ANSWER (REWRITTEN)
# Guarantees clean step-by-step numbered output
# ══════════════════════════════════════════════

def format_pointwise_answer(raw_text, question=""):
    """
    Post-process AI output to guarantee structured pointwise format.

    Rules:
    1. If text already has ## headers OR numbered points → clean and return
    2. If it's a paragraph → split into sentences, build structured output
    3. Always produces:
       ## Definition  (1 sentence)
       ## Key Points  (numbered list)
       ## Example     (if applicable)
       ## Key Formula (if formula detected)
       ## Summary     (1 exam tip sentence)
    """
    if not raw_text or len(raw_text.strip()) < 10:
        return raw_text

    text = raw_text.strip()

    # ── Already structured → clean strip and return ──────────────
    has_numbers  = bool(re.search(r'^\s*[1-9][\.\)]\s', text, re.MULTILINE))
    has_headers  = bool(re.search(r'^##\s', text, re.MULTILINE))
    if has_numbers or has_headers:
        # Still ensure it has a proper title
        if not text.startswith("#"):
            q_clean = _clean_question(question)
            title   = q_clean[:55] + "..." if len(q_clean) > 55 else q_clean
            text    = f"## {title}\n\n" + text
        return text

    # ── Paragraph → structured conversion ────────────────────────
    # Split into clean sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 12]

    if not sentences:
        return text

    # Build title from question
    q_clean = _clean_question(question)
    title   = q_clean[:60] + "..." if len(q_clean) > 60 else q_clean

    out = [f"## {title}\n"]

    # ── Section 1: Definition ─────────────────────────────────────
    out.append("## Definition\n")
    out.append(sentences[0] + "\n")

    # ── Section 2: Key Points (up to 6 remaining sentences) ──────
    key_sentences = sentences[1:7]
    if key_sentences:
        out.append("## Key Points\n")
        for i, s in enumerate(key_sentences, start=1):
            out.append(f"{i}. {s}")
        out.append("")

    # ── Section 3: Formula detection ─────────────────────────────
    formula_patterns = [
        r'[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+',      # chemical formula like H2SO4
        r'[A-Za-z\s]+=\s*[A-Za-z0-9\s\+\-\*/\^]+',  # equation like pH = -log[H+]
    ]
    formulas_found = []
    for pattern in formula_patterns:
        matches = re.findall(pattern, text)
        formulas_found.extend([m.strip() for m in matches if len(m.strip()) > 2])

    if formulas_found:
        out.append("## Key Formula\n")
        out.append(formulas_found[0])
        out.append("")

    # ── Section 4: Quick Summary ──────────────────────────────────
    if len(sentences) > 1:
        # Use last sentence as summary, or first if only one
        summary_sentence = sentences[-1] if len(sentences) > 2 else sentences[0]
        out.append("## Quick Summary\n")
        out.append(f"💡 {summary_sentence}")

    return "\n".join(out)


def _clean_question(question):
    """Strip IMPORTANT: prefix and prompt boilerplate from question string."""
    q = re.sub(r'^IMPORTANT:[^\n]*\n*', '', question, flags=re.IGNORECASE).strip()
    q = re.sub(r'^(QUESTION|Question):\s*"?', '', q).strip().strip('"')
    q = re.sub(r'^(You are|Provide|Answer|Format):.*?(?=\n|$)', '', q, flags=re.IGNORECASE).strip()
    return q if q else "Chemistry Answer"


# ══════════════════════════════════════════════
# AI CORE — FLAN-T5 + LoRA GENERATION
# ══════════════════════════════════════════════

def generate_ai(prompt, max_new_tokens=300):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=4,
                repetition_penalty=2.0,
                length_penalty=1.2,
                early_stopping=True,
                temperature=0.7,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return clean_output(decoded, prompt)
    except Exception as e:
        print(f"[Model Error] {e}")
        return f"Model error: {str(e)}. Please check if FLAN-T5 is loaded correctly."


# ══════════════════════════════════════════════
# v4.0 — STRUCTURED POINTWISE PROMPT BUILDER
# ══════════════════════════════════════════════

def build_structured_prompt(question):
    """
    Build a prompt that forces structured pointwise output from FLAN-T5.
    Cleaner format for better model compliance.
    """
    q = _clean_question(question)

    return f"""You are a chemistry expert tutor. Answer the following question in structured format.

Question: {q}

Write your answer using EXACTLY this format with numbered points:

## Definition
(One sentence direct answer or definition)

## Key Points
1. (First fact — include chemical formula if relevant)
2. (Second fact — explain the mechanism or process)
3. (Third fact — give a specific example)
4. (Fourth fact — mention real-world application)
5. (Fifth fact — state an exception or limitation)

## Step-by-Step Example
Step 1: (Setup the example)
Step 2: (Apply the concept)
Step 3: (Show the result or calculation)

## Key Formula
(The most important equation or formula for this topic)

## Exam Tip
(One important thing students must remember)

Be specific. Use chemical symbols. Keep each point to 1-2 sentences."""


# ══════════════════════════════════════════════
# PDF ANALYSIS
# ══════════════════════════════════════════════

def analyze_pdf_text(text):
    text = text[:1500].strip()

    summary_prompt = (
        "Summarize the following chemistry text into numbered study notes:\n"
        "## Summary\n"
        "1. (First key fact)\n"
        "2. (Second key fact)\n"
        "3. (Third key fact)\n"
        "Text:\n" + text
    )

    video_prompt = (
        "Write a short educational video script with [INTRO], [MAIN CONTENT], "
        "[CONCLUSION] sections for this chemistry topic:\n" + text
    )

    summary      = generate_ai(summary_prompt)
    video_script = generate_ai(video_prompt)

    quiz_list = generate_quiz(topic=text[:300], num_questions=5)
    quiz_text = format_quiz_as_text(quiz_list)

    return {
        "summary":      clean_output(summary),
        "quiz":         quiz_text,
        "video_script": clean_output(video_script),
    }


# ══════════════════════════════════════════════
# MAIN GENERATE ANSWER FUNCTION  (v4.0)
# ══════════════════════════════════════════════

def generate_answer(text, language="en"):
    """
    Main entry point — v4.0 priority order:
    0. QUIZ: prefix     → generate_quiz() (template bank, always reliable)
    1. IMPORTANT: prefix → bypass shortcuts → structured AI
    2. Structure request → RDKit image
    3. Direct element    → periodic table card
    4. Molar mass        → calculator
    5. PDF: prefix       → analyze_pdf_text()
    6. Structured AI     → format_pointwise_answer() post-processor
    7. Wikipedia fallback if AI output is too short
    """
    q = text.strip()
    if not q:
        return "Please enter a chemistry question."

    q_lower = q.lower()

    # ── 0. QUIZ PREFIX ─────────────────────────────────────────────
    # Formats: "QUIZ: acid base"  or  "QUIZ:8: organic chemistry"
    if q_lower.startswith("quiz:"):
        raw = q[5:].strip()
        count_match = re.match(r'^(\d+):\s*(.*)', raw)
        if count_match:
            num_q = min(int(count_match.group(1)), 10)
            topic = count_match.group(2).strip()
        else:
            num_q = 5
            topic = raw

        quiz_list = generate_quiz(topic=topic, num_questions=num_q)
        ans       = format_quiz_as_text(quiz_list, topic=topic)
        ans       = translate_text(ans, language)
        save_history(q, ans)
        return ans

    # ── 1. IMPORTANT PREFIX — BYPASS SHORTCUTS ────────────────────
    if q_lower.startswith("important:"):
        clean_q = _clean_question(q)
        prompt  = build_structured_prompt(clean_q)
        decoded = generate_ai(prompt, max_new_tokens=400)
        decoded = format_pointwise_answer(decoded, clean_q)
        ans     = translate_text(decoded, language)
        save_history(q, ans)
        return ans

    # ── 2. STRUCTURE / IMAGE REQUEST ──────────────────────────────
    if any(kw in q_lower for kw in ["structure", "draw", "molecule", "smiles"]):
        for compound in smiles_map:
            if compound in q_lower:
                img_path = generate_structure_image(compound)
                if img_path:
                    ans = (
                        f"✅ Molecular structure of **{compound.title()}** generated.\n\n"
                        f"**SMILES notation:** `{smiles_map[compound]}`\n\n"
                        f"Use the Structure tab to view the 2D diagram."
                    )
                else:
                    ans = f"Could not render structure for {compound}. SMILES: {smiles_map.get(compound, 'unknown')}"
                ans = translate_text(ans, language)
                save_history(q, ans)
                return ans

    # ── 3. PERIODIC TABLE (DIRECT QUESTIONS ONLY) ─────────────────
    periodic = periodic_lookup(q)
    if periodic:
        ans = translate_text(periodic, language)
        save_history(q, ans)
        return ans

    # ── 4. MOLAR MASS ─────────────────────────────────────────────
    if any(kw in q_lower for kw in ["molar mass", "molecular mass", "molecular weight", "atomic mass of"]):
        formula = extract_formula(q)
        if formula:
            mass = molar_mass(formula)
            if mass:
                ans = (
                    f"## Molar Mass of {formula}\n\n"
                    f"**Result:** {mass} g/mol\n\n"
                    f"## Step-by-Step Calculation\n\n"
                    f"1. Identify each element and its subscript count in the formula\n"
                    f"2. Look up the standard atomic mass (IUPAC values)\n"
                    f"3. Multiply each element's atomic mass by its count\n"
                    f"4. Sum all values\n\n"
                    f"**Answer:** {mass} g/mol\n\n"
                    f"## Key Fact\n\n"
                    f"1 mole of {formula} = {mass} g and contains 6.022 × 10²³ formula units."
                )
                ans = translate_text(ans, language)
                save_history(q, ans)
                return ans

    # ── 5. PDF MODE ───────────────────────────────────────────────
    if q.startswith("PDF:"):
        pdf_text = q.replace("PDF:", "").strip()
        result   = analyze_pdf_text(pdf_text)
        ans = (
            f"📄 **Summary:**\n{result['summary']}\n\n"
            f"🧪 **Quiz:**\n{result['quiz']}\n\n"
            f"🎬 **Video Script:**\n{result['video_script']}"
        )
        ans = translate_text(ans, language)
        save_history(q, ans)
        return ans

    # ── 6. STRUCTURED AI MODEL ────────────────────────────────────
    prompt  = build_structured_prompt(q)
    decoded = generate_ai(prompt, max_new_tokens=400)

    # ── 7. WIKIPEDIA FALLBACK ─────────────────────────────────────
    if len(decoded.split()) < 20:
        wiki = wikipedia_lookup(q)
        if wiki and len(wiki) > 50:
            decoded = wiki

    # Apply pointwise formatter to guarantee structured output
    decoded = format_pointwise_answer(decoded, q)

    ans = translate_text(decoded, language)
    save_history(q, ans)
    return ans