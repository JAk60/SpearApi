import ollama
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum, auto
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
    CATEGORY = auto()
    CRITICALITY = auto()
    LEVEL = auto()
    ENTITY = auto()
    FROM = auto()
    TASK_OBJECTIVE = auto()
    CONSTRAINTS = auto()
    OBJECTIVE = auto()


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


class NavalClassifier:
    def __init__(self, llm_model: str = "mistral", confidence_threshold: float = 0.20):
        """
        Initialize the Naval Classifier.

        Args:
            llm_model (str): Name of the Ollama model to use
            confidence_threshold (float): Threshold for confidence difference requiring clarification
        """
        self.llm_model = llm_model
        self.confidence_threshold = confidence_threshold
        self.key_to_layer = {
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
        self.multi_label_layers = {'Task Objective',
                                   'Constraints', 'Objective function'}

    def generate_question(self, layer_type: LayerType, top_class: str,
                          second_class: str, probability_diff: float) -> str:
        """Generate specific questions based on layer type and classes using ollama with naval context."""

        # Get context for the specific classes from NavalContext
        layer_context = NavalContext.LAYER_INDICATORS.get(layer_type, {})
        valid_classes = layer_context.get('classes', set())
        class_contexts = layer_context.get('context', {})

        top_class_context = class_contexts.get(top_class, [])
        second_class_context = class_contexts.get(second_class, [])

        prompt = f"""Task: Generate a specific yes/no question for naval operations classification.

Layer Type: {layer_type.name}
Primary Classification: {top_class}
Secondary Classification: {second_class}
Confidence Difference: {probability_diff:.1%}

Context Information:
- {top_class} characteristics: {', '.join(top_class_context) if top_class_context else 'No specific context available'}
- {second_class} characteristics: {', '.join(second_class_context) if second_class_context else 'No specific context available'}

Requirements:
1. Generate ONE yes/no question that clearly distinguishes {top_class} from {second_class}
2. Focus on the key differentiating characteristics from the context provided
3. The question must be answerable with yes/no
4. Use naval operational terminology where appropriate

Generate the question:"""

        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_tokens": 50
                }
            )
            question = response['response'].strip()
            return question if question.endswith('?') else question + '?'
        except Exception as e:
            return f"Is this specifically classified as {top_class} rather than {second_class}?"

    def process_results(self, model_output: Dict[str, Dict[str, float]],
                        save_csv: bool = True) -> List[Dict]:
        """
        Process naval classification results and optionally save to CSV.

        Args:
            model_output (Dict): Dictionary containing classification results
            save_csv (bool): Whether to save results to CSV

        Returns:
            List[Dict]: Processed classification results
        """
        layer_results = []

        for idx, (key, probabilities) in enumerate(model_output.items()):
            sorted_items = sorted(probabilities.items(),
                                  key=lambda x: x[1], reverse=True)
            layer_type = self.key_to_layer.get(key, LayerType.GENERAL)

            if key in self.multi_label_layers:
                top_prob = sorted_items[0][1]
                probable_labels = [(label, prob) for label, prob in sorted_items
                                   if (top_prob - prob) <= self.confidence_threshold]

                if len(probable_labels) > 1:
                    for i, (label1, prob1) in enumerate(probable_labels[:-1]):
                        for label2, prob2 in probable_labels[i+1:]:
                            layer_results.append({
                                'layer_number': idx + 1,
                                'layer_type': layer_type.name,
                                'class_1': label1,
                                'class_2': label2,
                                'prob_1': prob1,
                                'prob_2': prob2,
                                'prob_diff': prob1 - prob2,
                                'question': self.generate_question(
                                    layer_type, label1, label2, prob1 - prob2)
                            })
            else:
                top_class, top_prob = sorted_items[0]
                second_class, second_prob = sorted_items[1] if len(
                    sorted_items) > 1 else (None, 0)
                prob_diff = top_prob - second_prob

                if prob_diff < self.confidence_threshold:
                    layer_results.append({
                        'layer_number': idx + 1,
                        'layer_type': layer_type.name,
                        'class_1': top_class,
                        'class_2': second_class,
                        'prob_1': top_prob,
                        'prob_2': second_prob,
                        'prob_diff': prob_diff,
                        'question': self.generate_question(
                            layer_type, top_class, second_class, prob_diff)
                    })

        if save_csv:
            df = pd.DataFrame(layer_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_csv(
                f'Qlogs/naval_classification_results_{timestamp}.csv', index=False)

        return layer_results


# Example usage:
if __name__ == "__main__":
    # Sample model output
    model_output = {
  "category": {
    "Maintenance": 0.14524186437817183,
    "Mission": 0.8547581356218281
  },
  "sub category": {
    "Combat": 0.30377721910987837,
    "Exercise": 0.3911011321022979,
    "Fleet Support": 0.07534111459616466,
    "Humanitarian": 0.07931628104673391,
    "Miscellaneous": 0.09660859997803908,
    "Sortie": 0.053855653166886154
  },
  "criticality": { "High": 0.7319137458389157, "Low": 0.26808625416108434 },
  "Level": {
    "Equipment": 0.1362522069370461,
    "Fleet": 0.38136933892055713,
    "Ship": 0.4823784541423968
  },
  "Action": {
    "Evaluate": 0.6805260607413213,
    "Identify": 0.2585512738053958,
    "Select K out of N": 0.06092266545328303
  },
  "Entity": {
    "Equipment": 0.1305638271491649,
    "Ship": 0.8447906654021622,
    "Workshop": 0.024645507448672895
  },
  "From": {
    "Equipment": 0.06520324740741835,
    "Fleet": 0.5250044124437987,
    "Ships": 0.3871910237843727,
    "Workshops": 0.022601316364410463
  },
  "Task Objective": {
    "Gun firing": 0.16033171733045734,
    "Interrogation and interception": 0.48453786309249647,
    "Maintenance scheduling": 0.1508633031694188,
    "Miscellaneous": 0.03967037010855845,
    "Missile firing": 0.04449079965424319,
    "Search and rescue": 0.12010594664482586
  },
  "Constraints": {
    "Activity sequences": 0.09304600828092621,
    "Balancing loads": 0.22807498089426664,
    "Capability": 0.028190338086910488,
    "Conformance": 0.05433948801986134,
    "Endurance": 0.0495289686946764,
    "Fleet availability": 0.029137754553933498,
    "Fuel": 0.07147114039653106,
    "Logistic time": 0.060287873112099176,
    "Manpower availability": 0.02049111382651297,
    "Ration": 0.022130010175620187,
    "Reliability": 0.06327654395196583,
    "Risk score": 0.05078814276237799,
    "Ship class": 0.04755044386403331,
    "Spares availability": 0.050965677496238676,
    "Speed": 0.05018593593062285,
    "Working hours": 0.05378661086822928,
    "Workshop availability": 0.026748969085194144
  },
  "Objective function": {
    "Maximum availability": 0.13250017435178102,
    "Maximum conformance": 0.3821430594895493,
    "Maximum reliability": 0.03790231602701854,
    "Minimum cost": 0.20149710522408903,
    "Minimum downtime": 0.06796988715913868,
    "Minimum risk": 0.109952805603113,
    "Minimum time": 0.06803465214531028
  }
}


    # Create classifier instance
    classifier = NavalClassifier(
        llm_model="mistral", confidence_threshold=0.20)

    # Process results
    results = classifier.process_results(model_output)

    # Print results
    for result in results:
        print(f"\nLayer: {result['layer_type']}")
        print(f"Classes: {result['class_1']} vs {result['class_2']}")
        print(f"Question: {result['question']}")
