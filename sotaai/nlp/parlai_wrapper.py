# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''ParlAI's wrapper module.

Model information taken from:
https://github.com/facebookresearch/ParlAI/tree/master/parlai/agents
Dataset information taken from:
'''

MODELS = {
    'unknown': [
        'alice', 'bart', 'bert_classifier', 'bert_ranker', 'drqa',
        'fixed_response', 'hred', 'ir_baseline', 'local_human', 'memnn',
        'random_candidate', 'repeat_label', 'repeat_equery', 'retriever_reader',
        'safe_local_human', 'seq2seq', 'starspace', 'tfidf_retriever',
        'transformer', 'unigram'
    ]
}

DATASETS = {
    'unknown': [
        'airdialogue', 'amazon_qa', 'anli', 'aqua', 'babi',
        'blended_skill_talk', 'booktest', 'bot_adversarial_dialogue', 'c3',
        'cbt', 'ccpe', 'clevr', 'cnn_dm', 'coco_caption', 'commonsenseqa',
        'convai2', 'connvai2_wild_evaluation', 'convai_chitchat', 'copa',
        'coqa', 'cornell_movie', 'dailydialog', 'dbll_babi', 'dbll_movie',
        'dealnodeal', 'decanlp', 'decode', 'dialog_babi', 'dialog_babi_plus',
        'dialogue_nli', 'dialogue_qe', 'dialogue_safety', 'dream', 'dstc7',
        'eli5', 'empathetic_dialogues', 'flickr30k', 'fromfile', 'funpedia',
        'fvqa', 'genderation_bias', 'google_sgd', 'holl_e', 'hotpotqa', 'igc',
        'image_chat', 'insuranceqa', 'integration_tests', 'interactive',
        'iwslt14', 'jsonfile', 'light_dialog', 'light_dialog_wild',
        'light_genderation_bias', 'mctest', 'md_gender', 'mnist_qa',
        'moviedialog', 'ms_marco', 'mturkwikimovies', 'multinli',
        'multiwoz_v20', 'multiwoz_v21', 'mutualfriends', 'mwsc', 'narrative_qa',
        'natural_questions', 'nli', 'nlvr', 'onecommon', 'opensubtitles',
        'personachat', 'personality_captions', 'personalized_dialog', 'qacnn',
        'qadailymail', 'qangaroo', 'qasrl', 'qazre', 'quac', 'redial', 'scan',
        'self_chat', 'self_feeding', 'sensitive_topics_evaluation',
        'simplequestions', 'snli', 'squad', 'squad2', 'sst', 'style_gen',
        'taskmaster', 'taskmaster2', 'taskntalk', 'triviaqa', 'twitter',
        'ubuntu', 'visdial', 'vqa_v1', 'vqa_v2', 'webquestions', 'wikimovies',
        'wikipedia', 'wikiqa', 'wikisql', 'wizard_of_wikipedia', 'wmt', 'woz'
    ]
}
