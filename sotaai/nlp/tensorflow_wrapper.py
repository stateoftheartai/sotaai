# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Tensorflow Dataset's wrapper module.

Model information taken from:
Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'tensorflow',
    'original_name': 'TensorFlow Datasets',
    'url': 'https://www.tensorflow.org/datasets/'
}

DATASETS = {
    'question answering': [
        'cosmos_qa', 'mctaco', 'natural_questions', 'squad', 'trivia_qa/rc',
        'trivia_qa/rc.nocontext', 'trivia_qa/unfiltered',
        'trivia_qa/unfiltered.nocontext', 'web_questions'
    ],
    'structured': [
        'amazon_us_reviews/Wireless_v1_00', 'amazon_us_reviews/Watches_v1_00',
        'amazon_us_reviews/Video_Games_v1_00',
        'amazon_us_reviews/Video_DVD_v1_00', 'amazon_us_reviews/Video_v1_00',
        'amazon_us_reviews/Toys_v1_00', 'amazon_us_reviews/Tools_v1_00',
        'amazon_us_reviews/Sports_v1_00', 'amazon_us_reviews/Software_v1_00',
        'amazon_us_reviews/Shoes_v1_00', 'amazon_us_reviews/Pet_Products_v1_00',
        'amazon_us_reviews/Personal_Care_Appliances_v1_00',
        'amazon_us_reviews/PC_v1_00', 'amazon_us_reviews/Outdoors_v1_00',
        'amazon_us_reviews/Office_Products_v1_00',
        'amazon_us_reviews/Musical_Instruments_v1_00',
        'amazon_us_reviews/Music_v1_00',
        'amazon_us_reviews/Mobile_Electronics_v1_00',
        'amazon_us_reviews/Mobile_Apps_v1_00',
        'amazon_us_reviews/Major_Appliances_v1_00',
        'amazon_us_reviews/Luggage_v1_00',
        'amazon_us_reviews/Lawn_and_Garden_v1_00',
        'amazon_us_reviews/Kitchen_v1_00', 'amazon_us_reviews/Jewelry_v1_00',
        'amazon_us_reviews/Home_Improvement_v1_00',
        'amazon_us_reviews/Home_Entertainment_v1_00',
        'amazon_us_reviews/Home_v1_00',
        'amazon_us_reviews/Health_Personal_Care_v1_00',
        'amazon_us_reviews/Grocery_v1_00', 'amazon_us_reviews/Gift_Card_v1_00',
        'amazon_us_reviews/Furniture_v1_00',
        'amazon_us_reviews/Electronics_v1_00',
        'amazon_us_reviews/Digital_Video_Games_v1_00',
        'amazon_us_reviews/Digital_Video_Download_v1_00',
        'amazon_us_reviews/Digital_Software_v1_00',
        'amazon_us_reviews/Digital_Music_Purchase_v1_00',
        'amazon_us_reviews/Digital_Ebook_Purchase_v1_00',
        'amazon_us_reviews/Camera_v1_00', 'amazon_us_reviews/Books_v1_00',
        'amazon_us_reviews/Beauty_v1_00', 'amazon_us_reviews/Baby_v1_00',
        'amazon_us_reviews/Automotive_v1_00', 'amazon_us_reviews/Apparel_v1_00',
        'amazon_us_reviews/Digital_Ebook_Purchase_v1_01',
        'amazon_us_reviews/Books_v1_01', 'amazon_us_reviews/Books_v1_02'
    ],
    'summarization': [
        'aeslc', 'big_patent', 'billsum', 'cnn_dailymail', 'covid19sum',
        'gigaword', 'multi_news', 'newsroom',
        'opinion_abstracts/rotten_tomatoes', 'opinion_abstracts/idebate',
        'opinosis', 'reddit', 'reddit_tifu/short', 'reddit_tifu/long', 'samsum',
        'scientific_papers/arxiv', 'scientific_papers/pubmed', 'wikihow', 'xsum'
    ],
    'text classification': [
        'anli/r1', 'anli/r2', 'anli/r3', 'blimp/adjunct_island',
        'blimp/anaphor_gender_agreement', 'blimp/anaphor_number_agreement',
        'blimp/animate_subject_passive', 'blimp/animate_subject_trans',
        'blimp/causative', 'blimp/complex_NP_island',
        'blimp/coordinate_structure_constraint_complex_left_branch',
        'blimp/coordinate_structure_constraint_object_extraction',
        'blimp/determiner_noun_agreement_1',
        'blimp/determiner_noun_agreement_2',
        'blimp/determiner_noun_agreement_irregular_1',
        'blimp/determiner_noun_agreement_irregular_2',
        'blimp/determiner_noun_agreement_with_adj_2',
        'blimp/determiner_noun_agreement_with_adj_irregular_1',
        'blimp/determiner_noun_agreement_with_adj_irregular_2',
        'blimp/determiner_noun_agreement_with_adjective_1',
        'blimp/distractor_agreement_relational_noun',
        'blimp/distractor_agreement_relative_clause', 'blimp/drop_argument',
        'blimp/ellipsis_n_bar_1', 'blimp/ellipsis_n_bar_2',
        'blimp/existential_there_object_raising',
        'blimp/existential_there_quantifiers_1',
        'blimp/existential_there_quantifiers_2',
        'blimp/existential_there_subject_raising',
        'blimp/expletive_it_object_raising', 'blimp/inchoative',
        'blimp/intransitive', 'blimp/irregular_past_participle_adjectives',
        'blimp/irregular_past_participle_verbs',
        'blimp/irregular_plural_subject_verb_agreement_1',
        'blimp/irregular_plural_subject_verb_agreement_2',
        'blimp/left_branch_island_echo_question',
        'blimp/left_branch_island_simple_question',
        'blimp/matrix_question_npi_licensor_present', 'blimp/npi_present_1',
        'blimp/npi_present_2', 'blimp/only_npi_licensor_present',
        'blimp/only_npi_scope', 'blimp/passive_1', 'blimp/passive_2',
        'blimp/principle_A_c_command', 'blimp/principle_A_case_1',
        'blimp/principle_A_case_2', 'blimp/principle_A_domain_1',
        'blimp/principle_A_domain_2', 'blimp/principle_A_domain_3',
        'blimp/principle_A_reconstruction',
        'blimp/regular_plural_subject_verb_agreement_1',
        'blimp/regular_plural_subject_verb_agreement_2',
        'blimp/sentential_negation_npi_licensor_present',
        'blimp/sentential_negation_npi_scope',
        'blimp/sentential_subject_island', 'blimp/superlative_quantifiers_1',
        'blimp/superlative_quantifiers_2', 'blimp/tough_vs_raising_1',
        'blimp/tough_vs_raising_2', 'blimp/transitive', 'blimp/wh_island',
        'blimp/wh_questions_object_gap', 'blimp/wh_questions_subject_gap',
        'blimp/wh_questions_subject_gap_long_distance',
        'blimp/wh_vs_that_no_gap', 'blimp/wh_vs_that_no_gap_long_distance',
        'blimp/wh_vs_that_with_gap', 'blimp/wh_vs_that_with_gap_long_distance',
        'c4', 'cfq', 'civil_comments', 'clinic_oos', 'cos_e',
        'definite_pronoun_resolution', 'eraser_multi_rc', 'esnli', 'gap',
        'glue', 'goemotions', 'imdb_reviews', 'irc_disentaglement',
        'librispeech_lm', 'lm1b', 'math_dataset', 'movie_rationales',
        'multi_nli', 'multi_nli_mismatch', 'openbookqa', 'pg19', 'qa4mre',
        'reddit_disentanglement', 'scan', 'scicite', 'snli', 'super_glue',
        'tiny_shakespeare', 'wiki40b', 'wikipedia',
        'wikipedia_toxicity_subtypes', 'winogrande', 'wordnet', 'xnli',
        'yelp_popularity_reviews'
    ],
    'translation': [
        'flores', 'opus', 'para_crawl', 'ted_hrlr_translate',
        'ted_multi_translate', 'wmt14_translate', 'wmt15_translate',
        'wmt16_translate', 'wmt17_translate', 'wmt18_translate',
        'wmt19_translate', 'wmt_t2t_translate'
    ]
}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'tensorflow'}
