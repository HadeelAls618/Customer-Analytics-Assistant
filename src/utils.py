
from collections import defaultdict

def preprocess_rules(rules):
    rule_dict = defaultdict(list)
    for rule in rules:
        for lhs_item in rule.lhs:
            rule_dict[lhs_item].append((list(rule.rhs), rule.confidence, rule.lift))
    for lhs_item in rule_dict:
        rule_dict[lhs_item] = sorted(rule_dict[lhs_item], key=lambda x: (x[1], x[2]), reverse=True)
    return rule_dict
