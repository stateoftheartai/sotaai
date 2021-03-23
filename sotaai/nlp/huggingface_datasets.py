# -*- coding: utf-8 -*-
# Author: Eduardo Espinosa
# Copyright: Stateoftheart AI PBC 2021.
'''Hugging Face datsets dictionary module.'''

# Most of the models have too-long names that raises line-too-long warning, this is why
# pylint skips this file. Also, it is not relevant to lint this file since it
# only contains lists/dicts of data

# pylint: skip-file

DATASETS = {
    'machine-translation': [
        'alt', 'arxiv_dataset', 'autshumato', 'bianet', 'bible_para',
        'bsd_ja_en', 'capes', 'chr_en', 'ecb', 'eitb_parcc', 'emea',
        'europa_eac_tm', 'europa_ecdc_tm', 'flores', 'generated_reviews_enth',
        'giga_fren', 'hind_encorp', 'hkcancor', 'hrenwac_para', 'id_panl_bppt',
        'igbo_english_machine_translation', 'kde4', 'menyo20k_mt', 'mlsum',
        'ms_terms', 'msr_zhen_translation_parity', 'mt_eng_vietnamese',
        'multi_para_crawl', 'ncslgr', 'news_commentary', 'ofis_publik',
        'open_subtitles', 'opus_books', 'opus_dgt', 'opus_dogc', 'opus_elhuyar',
        'opus_euconst', 'opus_finlex', 'opus_fiskmo', 'opus_gnome',
        'opus_infopankki', 'opus_memat', 'opus_montenegrinsubs',
        'opus_openoffice', 'opus_paracrawl', 'opus_rf', 'opus_tedtalks',
        'opus_ubuntu', 'opus_wikipedia', 'opus_xhosanavy', 'php',
        'poleval2019_mt', 'qed_amara', 'scb_mt_enth_2020', 'scielo', 'setimes',
        'snow_simplified_japanese_corpus', 'spc', 'tanzil', 'tapaco', 'tatoeba',
        'ted_iwlst2013', 'ted_talks_iwslt', 'tep_en_fa_para', 'tilde_model',
        'udhr', 'um005', 'un_ga', 'un_multi', 'un_pc', 'wiki_source',
        'wiki_summary', 'wmt20_mlqe_task1', 'wmt20_mlqe_task2',
        'wmt20_mlqe_task3'
    ],
    'named-entity-recognition': [
        'afrikaans_ner_corpus', 'bc2gm_corpus', 'caner', 'conll2002',
        'conll2003', 'dane', 'ehealth_kd', 'euronews', 'finer',
        'german_legal_entity_recognition', 'germaner', 'harem', 'hausa_voa_ner',
        'id_nergrit_corpus', 'igbo_ner', 'indonlu', 'isixhosa_ner_corpus',
        'isizulu_ner_corpus', 'jnlpba', 'kor_ner', 'lener_br', 'limit',
        'linnaeus', 'lst20', 'msra_ner', 'NbAiLab/norne', 'ncbi_disease',
        'nchlt', 'nkjp-ner', 'norec', 'norwegian_ner', 'peoples_daily_ner',
        'persian_ner', 'ronec', 'sepedi_ner', 'sesotho_ner_corpus',
        'setswana_ner_corpus', 'siswati_ner_corpus', 'smartdata',
        'sofc_materials_articles', 'species_800', 'swedish_ner_corpus',
        'thainer', 'turkish_ner', 'turkish_shrinked_ner', 'turku_ner_corpus',
        'weibo_ner', 'wikiann', 'wino_bias', 'xglue', 'yoruba_gv_ner'
    ],
    'language-modeling': [
        'air_dialogue', 'amazon_reviews_multi', 'arabic_billion_words', 'brwac',
        'bswac', 'cawac', 'cc_news', 'cc100', 'chr_en', 'code_search_net',
        'cs_restaurants', 'dbrd', 'farsi_news', 'formermagic/github_python_1m',
        'hebrew_projectbenyehuda', 'hebrew_this_world', 'hrwac',
        'id_newspapers_2018', 'id_puisi', 'igbo_monolingual', 'makhzan', 'mkb',
        'newsph', 'opus100', 'oscar', 'para_pat', 'pib', 'ptb_text_only',
        'pubmed', 'recipe_nlg', 's2orc', 'sanskrit_classic', 'saudinewsnet',
        'silicone', 'spanish_billion_words', 'srwac', 'swahili', 'tashkeela',
        'telugu_books', 'telugu_news', 'thaisum', 'tlc', 'twi_text_c3',
        'wikicorpus', 'wikitext_tl39', 'yoruba_text_c3'
    ],
    'sentiment-classification': [
        'ajgt_twitter_ar', 'allocine', 'amazon_polarity',
        'amazon_reviews_multi', 'ar_res_reviews', 'ar_sarcasm', 'arsentd_lev',
        'cdt', 'dbrd', 'disaster_response_messages', 'dutch_social', 'ethos',
        'financial_phrasebank', 'flue', 'hate_speech_pl', 'hebrew_sentiment',
        'imdb_urdu_reviews', 'indonlu', 'muchocine', 'multi_booked',
        'NbAiLab/norec_agg', 'nsmc', 'oclar', 'omp', 'per_sent',
        'poem_sentiment', 'polemo2', 're_dial', 'senti_lex', 'silicone',
        'swedish_reviews', 'tamilmixsentiment', 'thai_toxicity_tweet', 'tsac',
        'tunizi', 'turkish_movie_sentiment', 'turkish_product_reviews',
        'tweet_eval', 'tweets_hate_speech_detection', 'urdu_sentiment_corpus',
        'wisesight_sentiment', 'wongnai_reviews', 'xed_en_fi',
        'yelp_review_full'
    ],
    'extractive-qa': [
        'adversarial_qa', 'aquamuse', 'covid_qa_castorini', 'covid_qa_deepset',
        'duorc', 'iapp_wiki_qa_squad', 'kilt_tasks', 'lhoestq/custom_squad',
        'liveqa', 'med_hop', 'mrqa', 'msr_sqa', 'multi_re_qa',
        'neural_code_search', 'newsqa', 'qed', 'quac', 'ropes', 'sharc',
        'sharc_modified', 'squad', 'squad_adversarial', 'squad_kor_v1',
        'squad_kor_v2', 'susumu2357/squad_v2_sv', 'thaiqa_squad', 'wiki_hop',
        'wiki_summary', 'xglue', 'xquad_r', 'zest'
    ],
    'multi-class-classification': [
        'circa', 'conceptnet5', 'danish_political_comments', 'dengue_filipino',
        'financial_phrasebank', 'generated_reviews_enth', 'go_emotions',
        'gutenberg_time', 'hard', 'hate_offensive', 'hate_speech_pl',
        'health_fact', 'indonlu', 'kinnews_kirnews', 'labr', 'limit', 'metooma',
        'mlsum', 'multi_woz_v22', 'onestop_english', 'pragmeval', 's2orc',
        'schema_guided_dstc8', 'swahili_news', 'telugu_news', 'tweet_eval',
        'universal_morphologies', 'woz_dialogue', 'xed_en_fi'
    ],
    'summarization': [
        'amazon_reviews_multi', 'arxiv_dataset', 'big_patent', 'billsum',
        'cnn_dailymail', 'gem', 'id_liputan6', 'mlsum', 'msr_text_compression',
        'multi_x_science_sum', 'orange_sum', 'pn_summary', 'polsum', 'psc',
        'recipe_nlg', 'samsum', 'scitldr', 'thaisum', 'wiki_asp',
        'wiki_atomic_edits', 'wiki_lingua', 'wiki_summary', 'xglue',
        'xsum_factuality'
    ],
    'open-domain-qa': [
        'adversarial_qa', 'ambig_qa', 'covid_qa_castorini', 'dyk', 'eli5',
        'freebase_qa', 'iapp_wiki_qa_squad', 'kilt_tasks', 'mkqa',
        'multi_re_qa', 'nq_open', 'proto_qa', 'qa_srl', 'selqa',
        'simple_questions_v2', 'so_stacksample', 'thaiqa_squad', 'tweet_qa',
        'wiki_qa_ar', 'wiki_summary', 'xglue', 'xor_tydi_qa', 'yahoo_answers_qa'
    ],
    'dialogue-modeling': [
        'air_dialogue', 'coached_conv_pref', 'craigslist_bargains',
        'cs_restaurants', 'curiosity_dialogs', 'dialog_re', 'gem', 'hkcancor',
        'irc_disentangle', 'kd_conv', 'kilt_tasks', 'meta_woz', 'multi_woz_v22',
        'mutual_friends', 'pec', 'quac', 'schema_guided_dstc8', 'silicone',
        'taskmaster1', 'taskmaster2', 'taskmaster3', 'woz_dialogue'
    ],
    'topic-classification': [
        'ag_news', 'amazon_reviews_multi', 'arsentd_lev', 'counter',
        'dbpedia_14', 'gnad10', 'hate_speech_pl', 'hausa_voa_topics',
        'kannada_news', 'kinnews_kirnews', 'mlsum', 'myanmar_news',
        'pn_summary', 'prachathai67k', 'pubmed', 'sofc_materials_articles',
        'telugu_news', 'xglue', 'yahoo_answers_topics', 'yoruba_bbc_topics'
    ],
    'intent-classification': [
        'bing_coronavirus_query_set', 'clinc_oos', 'diplomacy_detection',
        'disaster_response_messages', 'flue', 'hate_speech18', 'kor_3i4k',
        'kor_sae', 'poleval2019_cyberbullying', 'sms_spam',
        'snips_built_in_intents', 'tweet_eval', 'urdu_fake_news', 'xed_en_fi'
    ],
    'multi-label-classification': [
        'dutch_social', 'ethos', 'fake_news_english', 'go_emotions',
        'hate_speech_pl', 'jigsaw_toxicity_pred', 'kor_hate', 'metooma',
        'mlsum', 'ohsumed', 's2orc', 'swda', 'universal_morphologies',
        'xed_en_fi'
    ],
    'semantic-similarity-scoring': [
        'assin', 'assin2', 'climate_fever', 'counter', 'kor_nlu',
        'lavis-nlp/german_legal_sentences', 'paws', 'paws-x', 'refresd',
        'sem_eval_2014_task_1', 'stsb_mt_sv', 'twi_wordsim353',
        'yoruba_wordsim353'
    ],
    'fact-checking': [
        'ade_corpus_v2', 'clickbait_news_bg', 'climate_fever',
        'covid_tweets_japanese', 'datacommons_factcheck', 'factckbr',
        'fake_news_filipino', 'health_fact', 'id_clickbait', 'kilt_tasks',
        'tab_fact', 'urdu_fake_news'
    ],
    'multiple-choice-qa': [
        'aqua_rat', 'c3', 'codah', 'dream', 'exams', 'head_qa', 'mc_taco',
        'piqa', 'proto_qa', 'pubmed_qa', 'qa_srl', 'reasoning_bg'
    ],
    'natural-language-inference': [
        'assin', 'assin2', 'bbc_hindi_nli', 'hda_nli_hindi', 'imppres',
        'kor_nlu', 'newsph_nli', 'sem_eval_2014_task_1', 'sick', 'snli', 'swag',
        'xglue'
    ],
    'part-of-speech-tagging': [
        'arabic_pos_dialect', 'conll2002', 'conll2003', 'dane', 'indonlu',
        'lst20', 'mac_morpho', 'thainer', 'wikicorpus', 'wino_bias'
    ],
    'text-simplification': [
        'arxiv_dataset', 'asset', 'disaster_response_messages', 'gem',
        'onestop_english', 'pn_summary', 'times_of_india_news_headlines',
        'turk', 'wiki_auto', 'wiki_summary'
    ],
    'other-stuctured-to-text': [
        'atomic', 'cs_restaurants', 'enriched_web_nlg', 'gem', 'id_puisi',
        'nell', 'ollie', 'pubmed', 'times_of_india_news_headlines', 'web_nlg'
    ],
    'semantic-similarity-classification': [
        'flue', 'generated_reviews_enth', 'indonlu', 'kor_qpair',
        'medical_questions_pairs', 'paws', 'paws-x', 'refresd', 'tapaco',
        'wrbsc'
    ],
    'parsing': [
        'alt', 'amttl', 'coached_conv_pref', 'multi_woz_v22',
        'schema_guided_dstc8', 'web_nlg', 'woz_dialogue', 'xglue'
    ],
    'abstractive-qa': [
        'aquamuse', 'duorc', 'eli5', 'kilt_tasks', 'narrativeqa',
        'narrativeqa_manual', 'so_stacksample', 'wiki_summary'
    ],
    'fact-checking-retrieval': [
        'arxiv_dataset', 'climate_fever', 'evidence_infer_treatment', 'hover',
        'kilt_tasks', 'lama', 'nell', 'times_of_india_news_headlines'
    ],
    'sentiment-scoring': [
        'allegro_reviews', 'amazon_reviews_multi', 'app_reviews',
        'hate_speech_pl', 'oclar', 'senti_ws', 'turkish_movie_sentiment'
    ],
    'explanation-generation': [
        'arxiv_dataset', 'recipe_nlg', 'social_bias_frames',
        'times_of_india_news_headlines', 'wiki_atomic_edits', 'wiki_bio',
        'wiki_summary'
    ],
    'closed-domain-qa': [
        'covid_qa_deepset', 'covid_qa_ucsd', 'doc2dial', 'indonlu',
        'medical_dialog', 'wiki_movies', 'zest'
    ],
    'entity-linking-retrieval': [
        'arxiv_dataset', 'bprec', 'kilt_tasks', 'nell', 'recipe_nlg'
    ],
    'text-classification-other-hate-speech-detection': [
        'hate_offensive', 'hate_speech_offensive', 'hate_speech_portuguese',
        'hatexplain', 'offcombr'
    ],
    'slot-filling': [
        'formermagic/github_python_1m', 'kilt_tasks', 'numer_sense',
        'sofc_materials_articles', 'youtube_caption_corrections'
    ],
    'coreference-resolution': [
        'ade_corpus_v2', 'irc_disentangle', 'wino_bias', 'winograd_wsc'
    ],
    'document-retrieval': [
        'arxiv_dataset', 'kilt_tasks', 'recipe_nlg',
        'times_of_india_news_headlines'
    ],
    'table-to-text': ['gem', 'great_code', 'totto', 'wiki_bio'],
    'other-other-image-classification': ['cifar10', 'cifar100', 'mnist'],
    'text-scoring-other-evaluating-dialogue-systems': [
        'conv_ai', 'conv_ai_2', 'conv_ai_3'
    ],
    'conditional-text-generation-other-grammatical-error-correction': [
        'jfleg', 'tmu_gfm_dataset', 'wi_locness'
    ],
    'conditional-text-generation-other-dialogue-generation': [
        'air_dialogue', 'deal_or_no_dialog'
    ],
    'text-classification-other-sarcasm-detection': [
        'ar_sarcasm', 'kor_sarcasm'
    ],
    'other-other-automatic speech recognition': [
        'arabic_speech_corpus', 'librispeech_asr'
    ],
    'structure-prediction-other-word-tokenization': [
        'best2009', 'wisesight1000'
    ],
    'other-other-text-to-speech': ['crawl_domain', 'lj_speech'],
    'other-other-relation-extraction': ['dialog_re', 'ollie'],
    'conditional-text-generation-other-meaning-representtion-to-text': [
        'e2e_nlg', 'e2e_nlg_cleaned'
    ],
    'text-classification-other-language-identification': ['ilist', 'wili_2018'],
    'text_classification-other-news-category-classification': [
        'interpress_news_category_tr', 'ttc4900'
    ],
    'other-other-pretraining-language-models': [
        'large_spanish_corpus', 'spanish_billion_words'
    ],
    'question-answering-other-multi-hop': ['med_hop', 'wiki_hop'],
    'text-scoring-other-paraphrase-identification': ['paws', 'paws-x'],
    'question-answering-other-conversational-qa': ['sharc', 'sharc_modified'],
    'structure-prediction-other-acronym-identification': [
        'acronym_identification'
    ],
    'other-other-query-based-multi-document-summarization': ['aquamuse'],
    'other-other-data-mining': ['ar_cov19'],
    'text-scoring-other-simplification-evaluation': ['asset'],
    'text-classification-other-hate-speech-topic-classification': [
        'bn_hate_speech'
    ],
    'other-other-judgement-prediction---': ['cail2018'],
    'text-classification-other-stance-detection': ['catalonia_independence'],
    'other-other-sentences entailment and relatedness': ['cdsc'],
    'text-classification-other-question-answer-pair-classification': ['circa'],
    'other-other-Conversational Recommendation': ['coached_conv_pref'],
    'other-other-knowledge-extraction': ['cord19'],
    'other-other-speech-translation': ['covost2'],
    'other-other-web-search': ['crawl_domain'],
    'text-scoring-other-bias-evaluation': ['crows_pairs'],
    'sequence-modeling-other-conversational-curiosity': ['curiosity_dialogs'],
    'conditional-text-generation-other-rdf-to-text': ['dart'],
    'text-classification-other-discourse-marker-prediction': ['discovery'],
    'structure-prediction-other-relation-prediction': ['ehealth_kd'],
    'emotion-classification': ['emotone_ar'],
    'other-other-contextual-embeddings': ['eth_py150_open'],
    'text-classification-other-Hate Speech Detection': ['ethos'],
    'text-classification-other-Word Sense Disambiguation for Verbs': ['flue'],
    'other-other-knowledge-base': ['generics_kb'],
    'sequence-modeling-other-common-sense-inference': ['glucose'],
    'text-classification-other-emotion': ['go_emotions'],
    'other': ['google_wellformed_query'],
    'question-answering-other-knowledge-base-qa': ['grail_qa'],
    'text-scoring-other-Meronym-Prediction': ['has_part'],
    'sentiment-analysis': ['hate_speech_filipino'],
    'sequence-modeling-other-discourse-analysis': ['hindi_discourse'],
    'text-scoring-other-narrative-flow': ['hippocorpus'],
    'text-classification-other-hope-speech-classification': ['hope_edi'],
    'text-scoring-other-funniness-score-prediction': ['humicroedit'],
    'text-classification-other-funnier-headline-identification': [
        'humicroedit'
    ],
    'question-answering-other-multihop-tabular-text-qa': ['hybrid_qa'],
    'text-classification-other-aspect-based-sentiment-analysis': ['indonlu'],
    'structure-prediction-other-keyphrase-extraction': ['indonlu'],
    'structure-prediction-other-span-extraction': ['indonlu'],
    'conditional-text-generation-other-question-generation': ['inquisitive_qg'],
    'text-classification-other-question-identification': [
        'journalists_questions'
    ],
    'other-multi-turn': ['kd_conv'],
    'other-other-data-to-text-generation': ['kelm'],
    'text-scoring-other-probing': ['lama'],
    'conditional-text-generation-other-long-range-dependency': ['lambada'],
    'text-classification-other-fake-news-detection': ['liar'],
    'other-other-automatic-speech-recognition': ['lj_speech'],
    'structure-prediction-other-clause-segmentation': ['lst20'],
    'structure-prediction-other-sentence-segmentation': ['lst20'],
    'structure-prediction-other-word-segmentation': ['lst20'],
    'text-classification-other-gender-bias': ['md_gender_bias'],
    'other-other-disambiguation': ['medal'],
    'text-classification-other-poetry-classification': ['metrec'],
    'question-answering-other-generative-reading-comprehension-metric': [
        'mocha'
    ],
    'other-other-NCI-PID-PubMed Genomics Knowledge Base Completion Dataset': [
        'msr_genomics_kbcomp'
    ],
    'structure-prediction-other-fused-head-identification': [
        'numeric_fused_head'
    ],
    'text-classification-other-offensive-language-classification': [
        'offenseval2020_tr'
    ],
    'text-classification-other-offensive-language': ['offenseval_dravidian'],
    'utterance-retrieval': ['pec'],
    'text-classification-other-acceptability-classification': ['peer_read'],
    'text-scoring-other-citation-estimation': ['pubmed'],
    'sequence-modeling-code-modeling': ['py_ast'],
    'question-answering-other-explanations-in-question-answering': ['qed'],
    'text-classification-other-dialogue-sentiment-classification': ['re_dial'],
    'other-other-citation-recommendation': ['s2orc'],
    'text-classification-other-propaganda-technique-classification': [
        'sem_eval_2020_task_11'
    ],
    'structure-prediction-other-propaganda-span-identification': [
        'sem_eval_2020_task_11'
    ],
    'other-other-sentence-compression': ['sent_comp'],
    'structure-prediction-other-pos-tagging': ['senti_ws'],
    'text-classification-other-dialogue-act-classification': ['silicone'],
    'text-classification-other-emotion-classification': ['silicone'],
    'hate-speech-detection': ['social_bias_frames'],
    'conditional-text-generation-other-stuctured-to-text': ['spider'],
    'text-classification-other-stereotype-detection': ['stereoset'],
    'conditional-text-generation-other-given-a-sentence-generate-a-paraphrase-either-in-same-language-or-another-language':
        ['tapaco'],
    'other-diacritics-prediction': ['tashkeela'],
    'other-other-open-information-extraction': ['tuple_ie'],
    'other-other-machine-translation': ['tweets_ar_en_parallel'],
    'constituency-parsing': ['universal_dependencies'],
    'dependency-parsing': ['universal_dependencies'],
    'structure-prediction-other-morphology': ['universal_morphologies'],
    'conditional-text-generation': ['web_nlg'],
    'structure-prediction-other-lemmatization': ['wikicorpus'],
    'text-classification-other-word-sense-disambiguation': ['wikicorpus'],
    'text-classification-other-paraphrase identification': ['xglue'],
    'acceptability-classification': ['xglue'],
    'conditional-text-generation-other-question-answering': ['xglue'],
    'other-other-token-classification-of-text-errors': [
        'youtube_caption_corrections'
    ],
    'question-answering-other-yes-no-qa': ['zest'],
    'structure-prediction-other-output-structure': ['zest'],
    'code-generation': ['formermagic/github_python_1m'],
    'text-retrieval-other-example-based-retrieval': [
        'lavis-nlp/german_legal_sentences'
    ]
}
