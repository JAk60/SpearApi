import numpy as np
import re
import enum

import sys
sys.path.append('../../')

from spear.labeling import labeling_function, LFSet, ABSTAIN, preprocessor

from helper.con_scorer import word_similarity


class ClassLabels(enum.Enum):
    Maintenance = 0
    Mission = 1

THRESHOLD = 0.6

# Convert keywords in trigWord1 to lowercase
trigWord1 = {"fleet", "task force", "maritime operations", "deployment", "patrol", "exercise", "amphibious assault", "maritime security", "maneuvers", "fleet admiral", "base", "aviation", "seaborne operation", "vessel", "blockade", "warfare", "strategy", "surveillance", "convoy", "anti-submarine warfare", "combat", "mission objectives", "reconnaissance", "domain awareness", "presence", "drills", "escort", "fleet maneuvers", "operations center", "interception","mission","enemy","war", "mission,","mission's"}
trigWord2 = {"Repair", "Overhaul", "Refit", "Inspection", "Service", "Check-up", "Refurbishment", "Restoration", "Tune-up", "Fix", "Upgrade", "Restoration", "Refurbishment", "Inspection", "Overhaul", "Retrofit", "Revamp", "Refurbish", "Tune", "Lubrication", "Cleaning", "Calibration", "Testing", "Adjustment", "Replacement", "Painting", "Welding", "Greasing", "Polishing", "Troubleshooting","maintenance","annual","repair","restoration"}



@preprocessor()
def convert_to_lower(x):
    return x.lower().strip()


@labeling_function(resources=dict(keywords=trigWord1), pre=[convert_to_lower], label=ClassLabels.Mission)
def LF1(x, **kwargs):    
    if len(kwargs["keywords"].intersection(x.split())) > 0:
        return ClassLabels.Mission
    else:
        return ABSTAIN

@labeling_function(resources=dict(keywords=trigWord2), pre=[convert_to_lower], label=ClassLabels.Maintenance)
def LF2(x, **kwargs):
    if len(kwargs["keywords"].intersection(x.split())) > 0:
        return ClassLabels.Maintenance
    else:
        return ABSTAIN
    
@labeling_function(cont_scorer=word_similarity, resources=dict(keywords=trigWord1), pre=[convert_to_lower], label=ClassLabels.Mission)
def CLF1(c, **kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.Mission
    else:
        return ABSTAIN

@labeling_function(cont_scorer=word_similarity, resources=dict(keywords=trigWord2), pre=[convert_to_lower], label=ClassLabels.Maintenance)
def CLF2(c, **kwargs):
    if kwargs["continuous_score"] >= THRESHOLD:
        return ClassLabels.Maintenance
    else:
        return ABSTAIN



### use of regular expression for rule text

@labeling_function(pre=[convert_to_lower], label=ClassLabels.Mission)
def LF5(x):  
    pattern = r'\b([2-9][0-9]|[1-9][0-9]{2,}) nm\b' 
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Mission
    else:
        return ABSTAIN

@labeling_function(pre=[convert_to_lower], label=ClassLabels.Mission)
def LF6(x):  
    pattern = r'\b([4-9]|[1-9][0-9]{2,}) ac\b' 
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Mission
    else:
        return ABSTAIN
    
@labeling_function(pre=[convert_to_lower], label=ClassLabels.Mission)
def LF7(x):  
    pattern = r'[1-9]{3,} steering pumps' 
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Mission
    else:
        return ABSTAIN
    
@labeling_function(pre=[convert_to_lower], label=ClassLabels.Mission)    
def LF8(x):  
    pattern = r'stabiliser' 
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Mission
    else:
        return ABSTAIN

@labeling_function(pre=[convert_to_lower], label=ClassLabels.Mission)  
def LF9(x):  
    pattern = r'\b([0-9]|1[0-9]|2[0-9]) kw\b|power generation units on hot standby' 
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Mission
    else:
        return ABSTAIN
    
@labeling_function(pre=[convert_to_lower], label=ClassLabels.Mission)   
def LF10(x):  
    rpm_pattern = r'\b(1[5-9][0-9]|[2-9][0-9]{2,})\s*rpm\b'
    knots_pattern = r'\b(1[89]|[2-9][0-9]|[1-9][0-9]{2,})\s*knots\b'

    pattern = f'({rpm_pattern})|({knots_pattern})'

    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Mission
    else:
        return ABSTAIN



@labeling_function(pre=[convert_to_lower], label=ClassLabels.Maintenance)  
def LF11(x):  
    pattern = r'\b([0-9]|[1-4][0-9])% (das|diesel alternator)\b'
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Maintenance
    else:
        return ABSTAIN

@labeling_function(pre=[convert_to_lower], label=ClassLabels.Maintenance)  
def LF12(x):  
    pattern = r'Fire pumps on the ship are not in the ready state|Fire pumps available  0|fire pumps are chocked'
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Maintenance
    else:
        return ABSTAIN

@labeling_function(pre=[convert_to_lower], label=ClassLabels.Maintenance)  
def LF13(x):  
    pattern = r'\b helicopters onboard (1[5-9][0-9]|[2-9][1-9]{1,}) |helicopters are available for next 2 days|helo' 
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Maintenance
    else:
        return ABSTAIN


@labeling_function(pre=[convert_to_lower], label=ClassLabels.Maintenance)  
def LF14(x):  
    pattern = r'radar is not working | sonar is unavailable | sonar system needs to be changed| satelite communication' 
    match = re.search(pattern, x) 
    if match:
        return ClassLabels.Maintenance
    else:
        return ABSTAIN


LFS = [
    LF1,
    LF2,
    CLF1,
    CLF2,
    LF5,
    LF6,
    LF7,
    LF8,
    LF9,
    LF10,
    LF11,
    LF12,
    LF13,
    LF14
]

rules = LFSet("Category_LF")
rules.add_lf_list(LFS)
