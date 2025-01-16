import ollama
from typing import List, Dict, Tuple, Any, Generator
from dataclasses import dataclass
from enum import Enum, auto
import csv
from datetime import datetime
import pandas as pd


class LayerType(Enum):
    MISSION_TYPE = auto()
    PRIORITY = auto()
    ASSET = auto()
    ACTION = auto()
    ACTIVITY = auto()
    OPTIMIZATION = auto()
    RESOURCE = auto()
    GENERAL = auto()
    CATEGORY = auto()        # New
    CRITICALITY = auto()     # New
    LEVEL = auto()           # New
    ENTITY = auto()          # New
    FROM = auto()            # New
    TASK_OBJECTIVE = auto()  # New
    CONSTRAINTS = auto()      # New
    OBJECTIVE = auto()       # New


@dataclass
class NavalContext:
    """Enhanced context information for naval classification layers"""

    LAYER_INDICATORS = {
        LayerType.CATEGORY: {
            'classes': {'Maintenance', 'Mission'},
            'context': {
                'Maintenance': ["maintenance type", "scheduling priority", "resource requirements"],
                'Mission': ["mission objectives", "operational scope", "timeline"]
            }
        },
        LayerType.MISSION_TYPE: {
            'classes': {'Combat', 'Exercise', 'Fleet Support', 'Humanitarian', 'Sortie', 'Miscellaneous'},
            'context': {
                'Combat': ["threat level", "tactical situation", "combat readiness"],
                'Exercise': ["training objectives", "scenario complexity", "participating units"],
                'Fleet Support': ["support type", "resource requirements", "coordination needs"],
                'Humanitarian': ["situation urgency", "civilian needs", "resource availability"],
                'Sortie': ["mission objectives", "deployment timeline", "operational area"],
                'Miscellaneous': ["task nature", "general requirements", "basic needs"]
            }
        },
        LayerType.CRITICALITY: {
            'classes': {'High', 'Low'},
            'context': {
                'High': ["urgency level", "impact severity", "response time"],
                'Low': ["routine nature", "flexible timeline", "standard procedures"]
            }
        },
        LayerType.LEVEL: {
            'classes': {
                'Equipment',
                'Ship',
                'Squadron',
                'Fleet',
                'Task Force',
                'Battle Group',
                'Command'
            },
            'context': {
                'Equipment': [
                    "individual system operations",
                    "component-level maintenance",
                    "technical specifications",
                    "equipment-specific procedures"
                ],
                'Ship': [
                    "vessel-wide operations",
                    "ship-specific protocols",
                    "onboard resource management",
                    "ship system integration"
                ],
                'Squadron': [
                    "multiple ship coordination",
                    "tactical unit operations",
                    "squadron-level resources",
                    "formation management"
                ],
                'Fleet': [
                    "strategic deployment",
                    "multi-vessel coordination",
                    "theater-level operations",
                    "fleet-wide resource allocation"
                ],
                'Task Force': [
                    "mission-specific grouping",
                    "specialized operation coordination",
                    "cross-platform integration",
                    "task-oriented resource management"
                ],
                'Battle Group': [
                    "combat unit coordination",
                    "integrated defense operations",
                    "multi-capability management",
                    "battle group logistics"
                ],
                'Command': [
                    "strategic decision making",
                    "operational oversight",
                    "force-wide coordination",
                    "high-level resource planning"
                ]
            }
        },
        LayerType.ASSET: {
            'classes': {'Ship', 'Fleet', 'Equipment', 'Ships', 'Workshop', 'Workshops'},
            'context': {
                'Ship': ["vessel type", "operational status", "maintenance state"],
                'Fleet': ["fleet composition", "deployment status", "operational capacity"],
                'Equipment': ["equipment type", "functional status", "maintenance requirements"],
                'Ships': ["number of vessels", "fleet distribution", "operational status"],
                'Workshop': ["facility capability", "current capacity", "maintenance backlog"]
            }
        },
        LayerType.ACTION: {
            'classes': {'Evaluate', 'Identify', 'Select K out of N'},
            'context': {
                'Evaluate': ["assessment criteria", "performance metrics", "evaluation scope"],
                'Identify': ["identification parameters", "classification needs", "recognition factors"],
                'Select K out of N': ["selection criteria", "optimization goals", "constraints"]
            }
        },
        LayerType.ENTITY: {
            'classes': {'Equipment', 'Ship', 'Workshop'},
            'context': {
                'Equipment': ["equipment category", "operational status", "maintenance history"],
                'Ship': ["vessel class", "current mission", "readiness state"],
                'Workshop': ["facility type", "maintenance capability", "resource availability"]
            }
        },
        LayerType.FROM: {
            'classes': {'Equipment', 'Fleet', 'Ships', 'Workshops'},
            'context': {
                'Equipment': ["equipment source", "availability status", "deployment location"],
                'Fleet': ["fleet designation", "operational area", "command structure"],
                'Ships': ["vessel types", "deployment pattern", "operational readiness"],
                'Workshops': ["facility locations", "maintenance capacity", "resource status"]
            }
        },
        LayerType.TASK_OBJECTIVE: {
            'classes': {'Gun firing', 'Interrogation and interception', 'Maintenance scheduling',
                        'Missile firing', 'Search and rescue', 'Miscellaneous'},
            'context': {
                'Gun firing': ["target type", "ammunition status", "engagement rules"],
                'Interrogation and interception': ["contact classification", "response protocols", "tactical situation"],
                'Maintenance scheduling': ["maintenance priority", "resource availability", "operational impact"],
                'Missile firing': ["target assessment", "weapon readiness", "engagement criteria"],
                'Search and rescue': ["search area", "asset availability", "weather conditions"],
                'Miscellaneous': ["task specifics", "general requirements", "basic parameters"]
            }
        },
        LayerType.CONSTRAINTS: {
            'classes': {'Activity sequences', 'Balancing loads', 'Capability', 'Conformance', 'Endurance',
                        'Fleet availability', 'Fuel', 'Logistic time', 'Manpower availability', 'Ration',
                        'Reliability', 'Risk score', 'Ship class', 'Spares availability', 'Speed',
                        'Working hours', 'Workshop availability'},
            'context': {
                'Balancing loads': ["resource distribution", "operational demands", "capacity limits"],
                'Fuel': ["consumption rates", "resupply options", "operational requirements"],
                'Logistic time': ["delivery schedules", "supply chain status", "critical timelines"],
                'Activity sequences': ["operational flow", "dependency chains", "scheduling constraints"]
            }
        },
        LayerType.OBJECTIVE: {
            'classes': {'Maximum availability', 'Maximum conformance', 'Maximum reliability',
                        'Minimum cost', 'Minimum downtime', 'Minimum risk', 'Minimum time'},
            'context': {
                'Maximum availability': ["resource utilization", "operational readiness", "maintenance efficiency"],
                'Maximum conformance': ["compliance requirements", "standard adherence", "protocol following"],
                'Minimum cost': ["budget constraints", "resource optimization", "efficiency metrics"],
                'Minimum risk': ["risk assessment", "mitigation strategies", "safety protocols"]
            }
        }
    }


def generate_layer_specific_question(layer_type: LayerType, top_class: str, second_class: str,
                                     probability_diff: float) -> str:
    """Generate specific yes/no questions based on layer type and classes."""

    base_prompts = {
        LayerType.CATEGORY: f"""Generate a straightforward yes/no question to determine if this is a {top_class} task or a {second_class} task:
        Example: "Is this a {top_class.lower()} task or a {second_class.lower()} task?" """,

        LayerType.MISSION_TYPE: f"""Generate a straightforward yes/no question to determine if this is a {top_class} mission or a {second_class} mission:
        Example: "Is this a {top_class.lower()} mission or a {second_class.lower()} mission?" """,

        LayerType.CRITICALITY: f"""Generate a straightforward yes/no question to determine if this task has {top_class} criticality or {second_class} criticality:
        Example: "Does this task have {top_class.lower()} criticality or {second_class.lower()} criticality?" """,

        LayerType.LEVEL: f"""Generate a straightforward yes/no question to determine if this operates at the {top_class} level or the {second_class} level:
        Example: "Is this at the {top_class.lower()} level or the {second_class.lower()} level?" """,

        LayerType.ACTION: f"""Generate a straightforward yes/no question to determine if {top_class} action is required or {second_class} action:
        Example: "Does this require {top_class.lower()} action or {second_class.lower()} action?" """,

        LayerType.ENTITY: f"""Generate a straightforward yes/no question to determine if this is a {top_class} or a {second_class}:
        Example: "Is this a {top_class.lower()} or a {second_class.lower()}?" """,

        LayerType.FROM: f"""Generate a straightforward yes/no question to determine if this originates from {top_class} or {second_class}:
        Example: "Does this come from {top_class.lower()} or {second_class.lower()}?" """,

        LayerType.TASK_OBJECTIVE: f"""Generate a straightforward yes/no question to determine if this is a {top_class} or a {second_class}:
        Example: "Is {top_class.lower()} the primary objective or is it {second_class.lower()}?" """,

        LayerType.CONSTRAINTS: f"""Generate a straightforward yes/no question to determine if {top_class} is the main constraint or if it is {second_class}:
        Example: "Is {top_class.lower()} the main constraint or is it {second_class.lower()}?" """,

        LayerType.OBJECTIVE: f"""Generate a straightforward yes/no question to determine if {top_class} is the main objective or if it is {second_class}:
        Example: "Is {top_class.lower()} the main objective or is it {second_class.lower()}?" """
    }

    try:
        prompt = base_prompts.get(layer_type,
                                  f"Is this specifically classified as {top_class} rather than {second_class}?")

        response = ollama.generate(
            model="mistral",
            prompt=prompt,
            stream=False,
            options={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 50
            }
        )

        # Ensure response ends with a question mark and can be answered with yes/no
        question = response['response'].strip()
        if not question.endswith('?'):
            question += '?'
        return question
    except Exception as e:
        # Fallback yes/no questions based on layer type
        fallbacks = {
            LayerType.CATEGORY: f"Is this primarily a {top_class} task?",
            LayerType.MISSION_TYPE: f"Is this specifically a {top_class} mission?",
            LayerType.CRITICALITY: f"Is this a {top_class} priority task?",
            LayerType.ASSET: f"Is this classified as a {top_class}?",
            LayerType.ACTION: f"Does this require {top_class} action?",
            LayerType.ENTITY: f"Is this entity a {top_class}?",
            LayerType.FROM: f"Does this originate from {top_class}?",
            LayerType.TASK_OBJECTIVE: f"Is {top_class} the main objective?",
            LayerType.CONSTRAINTS: f"Is {top_class} the primary constraint?",
            LayerType.OBJECTIVE: f"Is {top_class} the main goal?"
        }
        return fallbacks.get(layer_type, f"Is this classified as {top_class}?")


def generate_layer_specific_question(layer_type: LayerType, top_class: str, second_class: str,
                                     probability_diff: float) -> str:
    """Generate specific questions based on layer type and classes using ollama."""

    prompts = {
        LayerType.CATEGORY: f"Generate one yes/no question to confirm if this is a {top_class} task that distinguishes it from {second_class}.",
        LayerType.MISSION_TYPE: f"Generate one yes/no question to confirm if this is a {top_class} mission that distinguishes it from {second_class}.",
        LayerType.CRITICALITY: f"Generate one yes/no question to confirm {top_class} criticality that distinguishes it from {second_class}.",
        LayerType.LEVEL: f"Generate one yes/no question to confirm if this operates at {top_class} level that distinguishes it from {second_class}.",
        LayerType.ACTION: f"Generate one yes/no question to confirm if {top_class} action is required that distinguishes it from {second_class}.",
        LayerType.ENTITY: f"Generate one yes/no question to confirm if this is a {top_class} that distinguishes it from {second_class}.",
        LayerType.FROM: f"Generate one yes/no question to confirm if this is from {top_class} that distinguishes it from {second_class}.",
        LayerType.TASK_OBJECTIVE: f"Generate one yes/no question to confirm if this is {top_class} that distinguishes it from {second_class}.",
        LayerType.CONSTRAINTS: f"Generate one yes/no question to confirm if {top_class} is the main constraint that distinguishes it from {second_class}.",
        LayerType.OBJECTIVE: f"Generate one yes/no question to confirm if {top_class} is the main objective that distinguishes it from {second_class}."
    }

    try:
        prompt = prompts.get(
            layer_type, f"Generate one yes/no question to confirm if this is {top_class} rather than {second_class}.")

        response = ollama.generate(
            model="mistral",
            prompt=prompt,
            stream=False,
            options={"temperature": 0.7, "top_p": 0.95, "max_tokens": 50}
        )

        question = response['response'].strip()
        return question if question.endswith('?') else question + '?'
    except Exception as e:
        fallbacks = {
            LayerType.CATEGORY: f"Is this primarily a {top_class} task?",
            LayerType.MISSION_TYPE: f"Is this specifically a {top_class} mission?",
            LayerType.CRITICALITY: f"Is this a {top_class} priority task?",
            LayerType.ASSET: f"Is this classified as a {top_class}?",
            LayerType.ACTION: f"Does this require {top_class} action?",
            LayerType.ENTITY: f"Is this entity a {top_class}?",
            LayerType.FROM: f"Does this originate from {top_class}?",
            LayerType.TASK_OBJECTIVE: f"Is {top_class} the main objective?",
            LayerType.CONSTRAINTS: f"Is {top_class} the primary constraint?",
            LayerType.OBJECTIVE: f"Is {top_class} the main goal?"
        }
        return fallbacks.get(layer_type, f"Is this classified as {top_class}?")


def save_to_csv(layer_results: List[Dict]) -> None:
    """Save results to CSV with multiple rows for multi-label classifications."""
    expanded_results = []

    for result in layer_results:
        if '|' in result['question']:
            # Split questions and create multiple rows
            questions = result['question'].split(' | ')
            for i in range(0, len(questions), 2):
                expanded_results.append({
                    'Layer_Number': result['layer_number'],
                    'Layer_Type': result['layer_type'],
                    'Class_1': result['class_1'],
                    'Class_2': result['class_2'],
                    'Class_1_Prob': result['prob_1'],
                    'Class_2_Prob': result['prob_2'],
                    'Probability_Difference': result['prob_diff'],
                    # Group questions in pairs
                    'Questions': ' | '.join(questions[i:i+2])
                })
        else:
            expanded_results.append({
                'Layer_Number': result['layer_number'],
                'Layer_Type': result['layer_type'],
                'Class_1': result['class_1'],
                'Class_2': result['class_2'],
                'Class_1_Prob': result['prob_1'],
                'Class_2_Prob': result['prob_2'],
                'Probability_Difference': result['prob_diff'],
                'Questions': result['question']
            })

    df = pd.DataFrame(expanded_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'naval_classification_results_{timestamp}.csv', index=False)


def filter_low_confidence_classifications(model_output: Dict[str, Dict[str, float]],
                                          confidence_threshold: float = 0.20) -> Dict[str, Dict[str, float]]:
    """
    Filter model outputs to only include classifications below the confidence threshold.

    Args:
        model_output: Dictionary of layer classifications and their probabilities
        confidence_threshold: Minimum probability difference required for confident classification

    Returns:
        Dictionary containing only the classifications that need clarification
    """
    filtered_output = {}

    for key, probabilities in model_output.items():
        sorted_items = sorted(probabilities.items(),
                              key=lambda x: x[1], reverse=True)

        if len(sorted_items) < 2:
            continue

        top_prob = sorted_items[0][1]

        # For multi-label layers, check if multiple classes are within threshold
        if key in {'Task Objective', 'Constraints', 'Objective function'}:
            close_probabilities = [(label, prob) for label, prob in sorted_items
                                   if (top_prob - prob) <= confidence_threshold]

            if len(close_probabilities) > 1:
                filtered_output[key] = {
                    label: prob for label, prob in close_probabilities}

        # For single-label layers, check difference between top two probabilities
        else:
            second_prob = sorted_items[1][1]
            if (top_prob - second_prob) < confidence_threshold:
                filtered_output[key] = {
                    sorted_items[0][0]: top_prob,
                    sorted_items[1][0]: second_prob
                }

    return filtered_output


def generate_classification_questions(model_output: Dict[str, Dict[str, float]],
                                      confidence_threshold: float = 0.20) -> Generator[Dict[str, Any], None, None]:
    """
    Generator function that yields one question at a time for a single object.

    Args:
        model_output: Dictionary containing a single layer classification and probabilities
        confidence_threshold: Minimum probability difference required for confident classification

    Yields:
        Dictionary containing current question information and metadata
    """
    key_to_layer = {
        'category': LayerType.CATEGORY,
        'sub category': LayerType.MISSION_TYPE,
        'criticality': LayerType.CRITICALITY,
        'Level': LayerType.LEVEL,
        'Action': LayerType.ACTION,
        'Entity': LayerType.ENTITY,
        'From': LayerType.FROM,
        'Task Objective': LayerType.TASK_OBJECTIVE,
        'Constraints': LayerType.CONSTRAINTS,
        'Objective function': LayerType.OBJECTIVE
    }

    # Process a single key-value pair
    key = list(model_output.keys())[0]  # Get the single key
    probabilities = model_output[key]    # Get its probabilities

    sorted_items = sorted(probabilities.items(),
                          key=lambda x: x[1], reverse=True)
    layer_type = key_to_layer.get(key, LayerType.GENERAL)
    top_prob = sorted_items[0][1]

    if key in {'Task Objective', 'Constraints', 'Objective function'}:
        # Get all labels within threshold
        close_probabilities = [(label, prob) for label, prob in sorted_items
                               if (top_prob - prob) <= confidence_threshold]

        if len(close_probabilities) > 1:
            # Generate one pair at a time
            for i, (label1, prob1) in enumerate(close_probabilities[:-1]):
                for label2, prob2 in close_probabilities[i+1:]:
                    question = generate_layer_specific_question(
                        layer_type, label1, label2, prob1 - prob2)

                    yield {
                        'layer_type': layer_type.name,
                        'layer_key': key,
                        'class_1': label1,
                        'class_2': label2,
                        'prob_1': prob1,
                        'prob_2': prob2,
                        'prob_diff': prob1 - prob2,
                        'question': question,
                        'is_multi_label': True
                    }

    else:
        # For single-label layers, check only top two if within threshold
        if len(sorted_items) > 1:
            top_class, top_prob = sorted_items[0]
            second_class, second_prob = sorted_items[1]
            prob_diff = top_prob - second_prob

            if prob_diff < confidence_threshold:
                question = generate_layer_specific_question(
                    layer_type, top_class, second_class, prob_diff)

                yield {
                    'layer_type': layer_type.name,
                    'layer_key': key,
                    'class_1': top_class,
                    'class_2': second_class,
                    'prob_1': top_prob,
                    'prob_2': second_prob,
                    'prob_diff': prob_diff,
                    'question': question,
                    'is_multi_label': False
                }
    # df = pd.DataFrame(layer_results)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # df.to_csv(f'naval_classification_results_{timestamp}.csv', index=False)


def process_naval_classification(low_confidence_outputs: Dict[str, Dict[str, float]],
                                 confidence_threshold: float = 0.20) -> list:
    """Process naval classification results, focusing only on low-confidence predictions."""

    key_to_layer = {
        'category': LayerType.CATEGORY,
        'sub category': LayerType.MISSION_TYPE,
        'criticality': LayerType.CRITICALITY,
        'Level': LayerType.LEVEL,
        'Action': LayerType.ACTION,
        'Entity': LayerType.ENTITY,
        'From': LayerType.FROM,
        'Task Objective': LayerType.TASK_OBJECTIVE,
        'Constraints': LayerType.CONSTRAINTS,
        'Objective function': LayerType.OBJECTIVE
    }

    multi_label_layers = {'Task Objective',
                          'Constraints', 'Objective function'}
    layer_results = []

    for idx, (key, probabilities) in enumerate(low_confidence_outputs.items()):
        sorted_items = sorted(probabilities.items(),
                              key=lambda x: x[1], reverse=True)
        layer_type = key_to_layer.get(key, LayerType.GENERAL)

        if key in multi_label_layers:
            # Generate questions for each pair of close predictions
            for i, (label1, prob1) in enumerate(sorted_items[:-1]):
                for label2, prob2 in sorted_items[i+1:]:
                    question = generate_layer_specific_question(
                        layer_type, label1, label2, prob1 - prob2)
                    layer_results.append({
                        'layer_number': idx + 1,
                        'layer_type': layer_type.name,
                        'class_1': label1,
                        'class_2': label2,
                        'prob_1': prob1,
                        'prob_2': prob2,
                        'prob_diff': prob1 - prob2,
                        'question': question
                    })
        else:
            # Generate question for top two predictions
            top_class, top_prob = sorted_items[0]
            second_class, second_prob = sorted_items[1]
            question = generate_layer_specific_question(
                layer_type, top_class, second_class, top_prob - second_prob)

            layer_results.append({
                'layer_number': idx + 1,
                'layer_type': layer_type.name,
                'class_1': top_class,
                'class_2': second_class,
                'prob_1': top_prob,
                'prob_2': second_prob,
                'prob_diff': top_prob - second_prob,
                'question': question
            })

    # Save results to CSV
    df = pd.DataFrame(layer_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f'naval_classification_results_{timestamp}.csv', index=False)

    return layer_results


# if __name__ == "__main__":
#     # model_output = {
#     #     "category": { "Maintenance": 0.17832720671875033, "Mission": 0.8216727932812498 },
#     #     "sub category": {
#     #     "Combat": 0.31016886980861963,
#     #     "Exercise": 0.4330184549171485,
#     #     "Fleet Support": 0.061985294129616474,
#     #     "Humanitarian": 0.06686852725328717,
#     #     "Miscellaneous": 0.0844734408327719,
#     #     "Sortie": 0.04348541305855628,
#     #     },
#     #     "criticality" :{ "High": 0.777929070105444, "Low": 0.222070929894556 },
#     #     "Level": {
#     #     "Equipment": 0.14413255177954573,
#     #     "Fleet": 0.35152918386900334,
#     #     "Ship": 0.504338264351451,
#     #     },
#     #     "Action" :{
#     #     "Evaluate": 0.7376144997714278,
#     #     "Identify": 0.22578907716778968,
#     #     "Select K out of N": 0.03659642306078251,
#     #     },
#     #     "Entity" :{
#     #     "Equipment": 0.11223792092220945,
#     #     "Ship": 0.8711500723830441,
#     #     "Workshop": 0.01661200669474646,
#     #     },
#     #     "From" :{
#     #     "Equipment": 0.06605544151264289,
#     #     "Fleet": 0.5105557222665112,
#     #     "Ships": 0.4042593767669045,
#     #     "Workshops": 0.0191294594539414,
#     #     },
#     #     "Task Objective" :{
#     #     "Gun firing": 0.15507889171794415,
#     #     "Interrogation and interception": 0.5417385344124299,
#     #     "Maintenance scheduling": 0.1322527435745234,
#     #     "Miscellaneous": 0.031025656794713703,
#     #     "Missile firing": 0.03430561641834795,
#     #     "Search and rescue": 0.10559855708204084,
#     #     },
#     #     "Constraints" :{
#     #     "Activity sequences": 0.12937414400053743,
#     #     "Balancing loads": 0.32565178870979816,
#     #     "Capability": 0.01804078786038079,
#     #     "Conformance": 0.04582248398232564,
#     #     "Endurance": 0.041880821331677136,
#     #     "Fleet availability": 0.020063459873457277,
#     #     "Fuel": 0.06180507852002175,
#     #     "Logistic time": 0.051940812839947385,
#     #     "Manpower availability": 0.012301981327424586,
#     #     "Ration": 0.012677641439303593,
#     #     "Reliability": 0.052638049395824976,
#     #     "Risk score": 0.03767696421282662,
#     #     "Ship class": 0.03807693940755162,
#     #     "Spares availability": 0.04232162694818574,
#     #     "Speed": 0.04838695631862371,
#     #     "Working hours": 0.04519670420682929,
#     #     "Workshop availability": 0.016143759625284283,
#     #     },
#     #     "Objective function" :{
#     #     "Maximum availability": 0.12942024794250806,
#     #     "Maximum conformance": 0.43786700006115337,
#     #     "Maximum reliability": 0.02886793100518615,
#     #     "Minimum cost": 0.18890363273256297,
#     #     "Minimum downtime": 0.058311411052936524,
#     #     "Minimum risk": 0.09927962375201765,
#     #     "Minimum time": 0.057350153453635325,
#     #     },
#     # }
#     # Sample input that will generate multiple questions for multi-label cases
#     model_output = {
#         'category': {'Mission': 0.82, 'Maintenance': 0.18},
#         'sub category': {'Exercise': 0.43, 'Combat': 0.31},
#         'criticality': {'High': 0.78, 'Low': 0.22},
#         'Level': {'Ship': 0.50, 'Fleet': 0.35},
#         'Action': {'Evaluate': 0.74, 'Identify': 0.23},
#         'Entity': {'Ship': 0.87, 'Equipment': 0.11},
#         'From': {'Fleet': 0.51, 'Ships': 0.40},
#         'Task Objective': {
#             'Interrogation': 0.54,
#             'Interception': 0.52,
#             'Navigation': 0.50,
#             'Communication': 0.48
#         },
#         'Constraints': {
#             'Balancing loads': 0.33,
#             'Activity sequences': 0.31,
#             'Time constraints': 0.30,
#             'Resource allocation': 0.29
#         },
#         'Objective function': {
#             'Maximum conformance': 0.44,
#             'Minimum deviation': 0.42,
#             'Optimal performance': 0.41,
#             'Cost efficiency': 0.40
#         }
#     }
# # Your existing model_output can be used directly since it's already in the correct format
# process_naval_classification(model_output)
