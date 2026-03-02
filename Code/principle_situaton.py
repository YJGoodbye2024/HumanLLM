# import pandas as pd
# import json

# 原则列表统一小写
sd_pri_list = ['actor observer asymmetry', 'defensive attribution hypothesis', 'effort justification', 'egocentric bias', 'false consensus effect', 'forer effect', 'fundamental attribution error', 'hard-easy effect', 'illusion of control', 'illusory superiority', 'optimism bias', 'overconfidence effect', 'risk compensation', 'self-serving bias', 'social desirability bias', 'third-person effect', 'decoy effect', 'reactance', 'social comparison bias', 'status quo bias', 'backfire effect', 'endowment effect', 'loss aversion', 'pseudocertainty effect', 'sunk cost fallacy', 'zero-risk bias', 'hyperbolic discounting', 'identifiable victim effect', 'ambiguity bias', 'belief bias', 'information bias', 'less-is-better effect', 'authority bias', 'automation bias', 'bandwagon effect', 'group attribution error', 'just-world hypothesis', 'stereotyping', 'ultimate attribution error', 'halo effect', 'in-group bias', 'out-group homogeneity bias', 'positivity effect', 'reactive devaluation', 'hindsight bias', 'impact bias', 'outcome bias', 'pessimism bias', 'planning fallacy', 'projection bias', 'restraint bias', 'self-consistency bias', 'denomination effect', 'mental accounting', 'normalcy bias', 'subadditivity effect', 'survivorship bias', 'zero-sum bias', 'anthropomorphism', 'illusion of validity', 'illusory correlation', 'curse of knowledge', 'illusion of asymmetric insight', 'illusion of transparency', 'spotlight effect', 'negativity bias', 'choice-supportive bias', 'confirmation bias', 'continued influence effect', 'expectation bias', 'observer effect', 'observer-expectancy effect', 'ostrich effect', 'bias blind spot', 'naive cynicism',
               'naive realism', 'attentional bias', 'availability heuristic', 'base rate fallacy', 'context effect', 'empathy gap', 'illusory truth effect', 'mere exposure effect', 'mood-congruent memory bias', 'omission bias', 'anchoring', 'conservatism', 'contrast effect', 'distinction bias', 'focusing effect', 'framing effect', 'fading affect bias', 'implicit association', 'implicit stereotypes', 'false memory', 'misattribution of memory', 'source confusion', 'misinformation effect', 'peak-end rule', 'delayed reciprocity', 'asymmetrical investment', 'survival imperative', 'aversion response', 'narrative self', 'hedonic adaptation', 'self-determination theory', 'pleasure principle & reality principle', 'search for meaning', 'moral licensing effect', 'choice overload', 'kin selection & inclusive fitness', 'asymmetrical parental investment', 'formation of dominance hierarchies', 'territoriality', 'mating strategies', 'jealousy', 'paternity uncertainty', 'groupthink', 'bystander effect', 'social facilitation', 'diffusion of responsibility', 'decision fatigue', 'awe', 'mortality salience & legacy drive', 'flow principle', 'gratitude mechanism', 'post-traumatic growth', 'skin hunger & the law of touch', 'self-handicapping paradox', 'the allure of the forbidden', 'sadistic pleasure', 'the utility principle of self-deception', 'play impulse principle', 'attribution theory', 'social comparison theory', 'self-perception theory', 'terror management theory', 'cognitive dissonance theory', 'psychological reactance theory', 'social learning theory', 'conformity', 'obedience to authority', 'social identity theory', 'reciprocity principle']

td_pri_list_100 = [
    # I. Extraversion (外倾性) - 20 words
    # Positive Pole
    'talkative', 'assertive', 'active', 'energetic', 'outgoing',
    'enthusiastic', 'daring', 'gregarious', 'bold', 'spontaneous',
    # Negative Pole
    'quiet', 'reserved', 'shy', 'inhibited', 'timid',
    'withdrawn', 'unassertive', 'introverted', 'silent', 'unenergetic',

    # II. Agreeableness (宜人性) - 20 words
    # Positive Pole
    'sympathetic', 'kind', 'appreciative', 'affectionate', 'soft-hearted',
    'warm', 'generous', 'trusting', 'helpful', 'cooperative',
    # Negative Pole
    'cold', 'unsympathetic', 'harsh', 'rude', 'unkind',
    'cruel', 'quarrelsome', 'critical', 'antagonistic', 'callous',

    # III. Conscientiousness (尽责性) - 20 words
    # Positive Pole
    'organized', 'responsible', 'dependable', 'thorough', 'efficient',
    'practical', 'deliberate', 'conscientious', 'neat', 'careful',
    # Negative Pole
    'disorganized', 'careless', 'irresponsible', 'undependable', 'sloppy',
    'impractical', 'haphazard', 'negligent', 'untidy', 'rash',

    # IV. Emotional Stability (情绪稳定性) - 20 words
    # Positive Pole
    'relaxed', 'calm', 'at ease', 'unemotional', 'poised',
    'composed', 'secure', 'stable', 'content', 'placid',
    # Negative Pole (Neuroticism)
    'anxious', 'moody', 'envious', 'touchy', 'fretful',
    'temperamental', 'insecure', 'nervous', 'jealous', 'high-strung',

    # V. Openness (开放性) - 20 words
    # Positive Pole
    'creative', 'imaginative', 'intellectual', 'philosophical', 'complex',
    'deep', 'artistic', 'bright', 'perceptive', 'introspective',
    # Negative Pole
    'uncreative', 'unimaginative', 'unintellectual', 'unphilosophical', 'simple',
    'shallow', 'unartistic', 'dull', 'imperceptive', 'uninquisitive'
]


Situation_list = [
    "Duty: A situation that is work- or task-oriented.",
    "Intellect: A situation that requires intellectual engagement and reasoning.",
    "Adversity: A situation that involves threats or criticism.",
    "Mating: A situation that involves a potential romantic relationship.",
    "Positivity: A situation that is fun and enjoyable.",
    "Negativity: A situation that can trigger negative emotions.",
    "Deception: A situation where there is a possibility of distrust and deception.",
    "Sociality: A situation that requires social interaction."
]


Situation_list_2 = [
    "Home & Private Life",
    "Education & Academic",
    "Leisure & Socializing",
    "Services & Transactions",
    "Health & Well-being",
    "Romantic & Intimate",
]
# excel_file = './Dataset/principle.xlsx'


# def ptoe():
#     df = pd.DataFrame(pri_list)
#     df.to_excel(excel_file, index=False, header=False)
#     return


# def etop():
#     df = pd.read_excel(excel_file, header=None)
#     pri_list = df[0].tolist()
#     print(pri_list)
#     return


# if __name__ == '__main__':
# ptoe()
# etop()
# print(len(pri_list))
# print(f"Total items in the list: {len(situation)}")
# print(situation_domain)
# print(situation_domain)
# print(len(Situation))

# with open('./Dataset/Personal Traits.txt', 'r', encoding='utf-8') as f:
#     personality_traits = [line.strip()
#                           for line in f.readlines() if line.strip()]
# with open('./Dataset/traits.json', 'w', encoding='utf-8') as fw:
#     json.dump(personality_traits, fw, indent=2, ensure_ascii=False)
