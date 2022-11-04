import argparse
import bleu2
import weighted_ngram_match
import syntax_match
import dataflow_match
import os

def calc_code_bleu(ref, hyp, lang, keywords_dir):
    
    if type(ref)==str:
        pre_references = [[x.strip() for x in open(ref, 'r', encoding='utf-8').readlines()]]
    else:
        pre_references = ref
    if type(hyp)==str:
        hypothesis = [x.strip() for x in open(hyp, 'r', encoding='utf-8').readlines()]
    else:
        hypothesis = hyp
    
    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references)*len(hypothesis)


    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu2.corpus_bleu(tokenized_refs,tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open(os.path.join(keywords_dir,lang+'.txt'), 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(references, hypothesis, lang)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(references, hypothesis, lang)

    code_bleu_score = (ngram_match_score\
                        + weighted_ngram_match_score\
                        + syntax_match_score\
                        + dataflow_match_score)/4
    return [ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score, code_bleu_score]


def calc_code_bleu_multilang(ref, hyp, langs, keywords_dir):
    
    pre_references = [[x.strip() for x in open(ref, 'r', encoding='utf-8').readlines()]]
    hypothesis = [x.strip() for x in open(hyp, 'r', encoding='utf-8').readlines()]
    
    ret = {}
    lang_set = set(langs)
    for lang in lang_set:
        ind = [i for i in range(len(langs)) if langs[i]==lang]
        ret[lang] = calc_code_bleu([[pre_references[0][i] for i in ind]], \
                                   [hypothesis[i] for i in ind], \
                                   lang, keywords_dir)
    return ret
    
    