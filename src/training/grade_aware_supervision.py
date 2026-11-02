"""Conservative report-size adjustment; this extracts text, not image truth.

Only explicit English small/trace effusion or Baker's cyst wording is eligible.
Higher or mixed sizes, uncertain statements and historical sections abstain.
The probability ceiling is fixed before expert image validation.
"""
import re

POLICY_VERSION = 'english_small_fluid_v1'
LOW_PROBABILITY_CEILING = 0.15
LOW = r'(?:trace|minimal|small|mild|tiny)'
HIGH = r'(?:moderate|large|marked|massive|significant|substantial|severe)'
LINK = r'(?:\s+(?:amount|of|a|an|the|left|right|knee|joint|popliteal|leaking|intact|unruptured)){0,7}\s+'
NOUNS = {'Effusion': r'(?:(?:knee\s+)?(?:joint\s+)?effusion|joint\s+fluid)',
         "Baker's": r'(?:Baker[’\x27]?s?\s+cyst|popliteal\s+cyst)'}
UNCERTAIN = re.compile(r'\b(?:possible|possibly|suspected|questionable|cannot\s+exclude|rule\s+out|r/o)\b', re.I)
HISTORY = re.compile(r'\b(?:history|historical|previously|prior|comparison|indication|clinical\s+information)\b',re.I)
NEGATED = re.compile(r'\b(?:no|without|absent|negative\s+for)\b',re.I)


def evidence(report, target):
    if target not in NOUNS:
        return {'state':'OUT_OF_SCOPE','low_spans':[],'high_spans':[]}
    noun=NOUNS[target]
    low=re.compile(r'\b'+LOW+LINK+noun+r'\b',re.I)
    high=re.compile(r'\b'+HIGH+LINK+noun+r'\b',re.I)
    lows,highs,abstained=[],[],[]
    # Paragraph headers are retained, so clinical-history text can abstain.
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', report):
        matches=[m for m in low.finditer(sentence)]
        higher=[m for m in high.finditer(sentence)]
        if not matches and not higher:
            continue
        if HISTORY.search(sentence) or UNCERTAIN.search(sentence):
            abstained.append(sentence.strip());continue
        if NEGATED.search(sentence):
            # Conservative: a negation elsewhere in the sentence also abstains.
            abstained.append(sentence.strip());continue
        lows.extend(m.group(0) for m in matches)
        highs.extend(m.group(0) for m in higher)
    if highs or abstained:
        state='ABSTAIN_HIGH_OR_AMBIGUOUS'
    elif lows:
        state='EXPLICIT_LOW_SIZE'
    else:
        state='NO_ELIGIBLE_WORDING'
    return {'state':state,'low_spans':lows,'high_spans':highs,'abstained_sentences':abstained}


def adjusted_probability(report, target, probability):
    if not 0 <= probability <= 1:
        raise ValueError('invalid control probability')
    finding=evidence(report,target)
    candidate=min(probability,LOW_PROBABILITY_CEILING) if finding['state']=='EXPLICIT_LOW_SIZE' else probability
    return candidate,finding
