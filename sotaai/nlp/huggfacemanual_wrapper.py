# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Hugging Face's Library wrapper.

Model information taken from:
Dataset information taken from:
'''

SOURCE_METADATA = {
    'name': 'huggingface',
    'original_name': 'Hugging Face',
    'url': 'https://huggingface.co/'
}

MODELS = {
    'Conditional Generation': [
        'BART', 'Blenderbot', 'FSMT', 'LED', 'M2M100', 'MBart', 'MBart-50',
        'MT5', 'Pegasus', 'ProphetNet', 'Speech2Text', 'T5'
    ],
    'Next Sentence Prediction': ['BERT', 'BORT', 'MobileBERT'],
    'Language Modeling': [
        'ALBERT', 'BART', 'BERT', 'BigBird', 'BORT', 'CamemBERT', 'ConvBERT',
        'DeBERTa', 'DeBERTa-v2', 'DistilBERT', 'ELECTRA', 'Funnel Transformer',
        'LayoutLM', 'Longformer', 'MobileBERT', 'MPNet', 'OpenAI GPT',
        'OpenAI GPT2', 'Reformer', 'RetriBERT', 'RoBERTa', 'SqueezeBERT',
        'TAPAS', 'Transformer XL', 'XLM', 'XLM-RoBERTa', 'XLNet'
    ],
    'Causal Language Modeling': [
        'BART', 'BigBird', 'Blenderbot', 'CamemBERT', 'MarianMT', 'GPT Neo',
        'Pegasus', 'ProphetNet', 'RoBERTa', 'XLM-RoBERTa'
    ],
    'Sequence Classification': [
        'ALBERT', 'BART', 'BERT', 'BigBird', 'BORT', 'CamemBERT', 'ConvBERT',
        'CTRL', 'DeBERTa', 'DeBERTa-v2', 'DistilBERT', 'ELECTRA', 'FlauBERT',
        'Funnel Transformer', 'I-BERT', 'LayoutLM', 'LED', 'Longformer',
        'MBart', 'MBart-50', 'MobileBERT', 'MPNet', 'OpenAI GPT', 'OpenAI GPT2',
        'Reformer', 'RoBERTa', 'SqueezeBERT', 'TAPAS', 'Transformer XL', 'XLM',
        'XLM-RoBERTa', 'XLNet'
    ],
    'Token Classification': [
        'ALBERT', 'BERT', 'BigBird', 'BORT', 'CamemBERT', 'ConvBERT', 'DeBERTa',
        'DeBERTa-v2', 'DistilBERT', 'ELECTRA', 'FlauBERT', 'Funnel Transformer',
        'I-BERT', 'LayoutLM', 'Longformer', 'MobileBERT', 'MPNet', 'RoBERTa',
        'SqueezeBERT', 'XLM', 'XLM-RoBERTa', 'XLNet'
    ],
    'Question Answering': [
        'ALBERT', 'BART', 'BERT', 'BigBird', 'BORT', 'CamemBERT', 'ConvBERT',
        'DeBERTa', 'DeBERTa-v2', 'DistilBERT', 'ELECTRA', 'FlauBERT',
        'Funnel Transformer', 'I-BERT', 'LED', 'Longformer', 'LXMERT',
        'MobileBERT', 'MBart', 'MBart-50', 'MobileBERT', 'MPNet', 'Reformer',
        'RoBERTa', 'SqueezeBERT', 'TAPAS', 'XLM', 'XLM-ProphetNet',
        'XLM-RoBERTa', 'XLNet'
    ],
    'Text Classification': ['Bertweet',],
    'Part-of-speech Tagging': ['Bertweet',],
    'Named Entity Recognition': ['Bertweet',],
    'Summarization': ['BARThez', 'BertGeneration'],
    'Text Generation': ['DialoGPT', 'RAG', 'XLM-ProphetNet'],
    'Token Generation': ['RAG',],
    'Machine Translation': ['BertGeneration',],
    'Sentence Splitting': ['BertGeneration',],
    'Sentence Fusion': ['BertGeneration',],
    'Multilingual': ['herBERT', 'PhoBERT'],
    'Image Classification': ['Vision Transformer (ViT)',],
    'Speech Recognition': ['Wav2Vec2', 'XLSR-Wav2Vec2'],
}

DATASETS = {
    'text-classification': [
        'ade_corpus_v2', 'ag_news', 'ajgt_twitter_ar', 'allocine',
        'amazon_polarity', 'amazon_reviews_multi', 'ar_res_reviews',
        'ar_sarcasm', 'arsentd_lev', 'assin', 'assin2', 'banking77',
        'bbc_hindi_nli', 'bing_coronavirus_query_set', 'bn_hate_speech',
        'catalonia_independence', 'cdt', 'circa', 'clickbait_news_bg',
        'climate_fever', 'clinc_oos', 'conceptnet5', 'counter',
        'covid_tweets_japanese', 'danish_political_comments',
        'datacommons_factcheck', 'dbpedia_14', 'dbrd', 'dengue_filipino',
        'diplomacy_detection', 'disaster_response_messages', 'discovery',
        'dutch_social', 'ecthr_cases', 'EMBO/sd-nlp', 'emotone_ar', 'ethos',
        'eurlex', 'factckbr', 'fake_news_english', 'fake_news_filipino',
        'financial_phrasebank', 'flue', 'generated_reviews_enth', 'gnad10',
        'go_emotions', 'gutenberg_time', 'hard', 'hate_offensive',
        'hate_speech_filipino', 'hate_speech_offensive', 'hate_speech_pl',
        'hate_speech_portuguese', 'hate_speech18', 'hatexplain',
        'hausa_voa_topics', 'hda_nli_hindi', 'health_fact', 'hebrew_sentiment',
        'hope_edi', 'humicroedit', 'id_clickbait', 'ilist', 'imdb_urdu_reviews',
        'imppres', 'indonlu', 'interpress_news_category_tr',
        'interpress_news_category_tr_lite', 'jigsaw_toxicity_pred',
        'journalists_questions', 'kannada_news', 'kilt_tasks',
        'kinnews_kirnews', 'kor_3i4k', 'kor_hate', 'kor_nlu', 'kor_qpair',
        'kor_sae', 'kor_sarcasm', 'labr', 'laroseda', 'liar', 'limit',
        'md_gender_bias', 'medical_questions_pairs', 'metooma', 'metrec',
        'miam', 'mlsum', 'moroco', 'muchocine', 'multi_booked', 'multi_woz_v22',
        'myanmar_news', 'NbAiLab/norec_agg', 'NbAiLab/norwegian_parliament',
        'newsph_nli', 'nsmc', 'oclar', 'offcombr', 'offenseval_dravidian',
        'offenseval2020_tr', 'ohsumed', 'omp', 'onestop_english', 'paws',
        'paws-x', 'peer_read', 'peixian/equity_evaluation_corpus',
        'peixian/rtGender', 'per_sent', 'pn_summary', 'poem_sentiment',
        'polemo2', 'poleval2019_cyberbullying', 'prachathai67k', 'pragmeval',
        'pubmed', 're_dial', 'refresd', 'ro_sent', 's2orc',
        'schema_guided_dstc8', 'sem_eval_2014_task_1', 'sem_eval_2020_task_11',
        'senti_lex', 'sick', 'silicone', 'sms_spam', 'snips_built_in_intents',
        'snli', 'social_bias_frames', 'sofc_materials_articles', 'sst',
        'stereoset', 'swag', 'swahili_news', 'swda', 'swedish_reviews',
        'tab_fact', 'tamilmixsentiment', 'tapaco', 'telugu_news',
        'thai_toxicity_tweet', 'tsac', 'ttc4900', 'tunizi',
        'turkish_movie_sentiment', 'turkish_product_reviews', 'tweet_eval',
        'tweets_hate_speech_detection', 'universal_morphologies',
        'urdu_fake_news', 'urdu_sentiment_corpus', 'w11wo/imdb-javanese',
        'wikicorpus', 'wili_2018', 'wisesight_sentiment', 'wongnai_reviews',
        'woz_dialogue', 'wrbsc', 'xed_en_fi', 'xglue', 'yahoo_answers_topics',
        'yelp_review_full', 'yoruba_bbc_topics'
    ],
    'conditional-text-generation': [
        'air_dialogue', 'alt', 'amazon_reviews_multi', 'arxiv_dataset', 'asset',
        'atomic', 'autshumato', 'bianet', 'bible_para', 'big_patent', 'billsum',
        'bsd_ja_en', 'capes', 'chr_en', 'cnn_dailymail', 'conv_ai', 'conv_ai_2',
        'conv_ai_3', 'cs_restaurants', 'dart', 'deal_or_no_dialog',
        'disaster_response_messages', 'e2e_nlg', 'e2e_nlg_cleaned', 'ecb',
        'eitb_parcc', 'emea', 'enriched_web_nlg', 'europa_eac_tm',
        'europa_ecdc_tm', 'flores', 'formermagic/github_python_1m', 'gem',
        'generated_reviews_enth', 'giga_fren', 'great_code', 'hind_encorp',
        'hkcancor', 'hrenwac_para', 'id_liputan6', 'id_panl_bppt', 'id_puisi',
        'igbo_english_machine_translation', 'inquisitive_qg', 'jfleg', 'kde4',
        'lambada', 'menyo20k_mt', 'mlsum', 'ms_terms', 'msr_text_compression',
        'msr_zhen_translation_parity', 'mt_eng_vietnamese', 'multi_para_crawl',
        'multi_x_science_sum', 'ncslgr', 'nell', 'news_commentary',
        'ofis_publik', 'onestop_english', 'open_subtitles', 'opus_books',
        'opus_dgt', 'opus_dogc', 'opus_elhuyar', 'opus_euconst', 'opus_finlex',
        'opus_fiskmo', 'opus_gnome', 'opus_infopankki', 'opus_memat',
        'opus_montenegrinsubs', 'opus_openoffice', 'opus_paracrawl', 'opus_rf',
        'opus_tedtalks', 'opus_ubuntu', 'opus_wikipedia', 'opus_xhosanavy',
        'orange_sum', 'php', 'pn_summary', 'poleval2019_mt', 'polsum', 'psc',
        'pubmed', 'qed_amara', 'recipe_nlg', 'ro_sts_parallel', 'samsum',
        'scb_mt_enth_2020', 'scielo', 'scitldr', 'setimes',
        'snow_simplified_japanese_corpus', 'social_bias_frames', 'spc',
        'spider', 'tanzil', 'tapaco', 'tatoeba', 'ted_iwlst2013',
        'ted_talks_iwslt', 'tep_en_fa_para', 'thaisum', 'tilde_model',
        'times_of_india_news_headlines', 'tmu_gfm_dataset', 'totto', 'turk',
        'udhr', 'um005', 'un_ga', 'un_multi', 'un_pc', 'wi_locness', 'wiki_asp',
        'wiki_atomic_edits', 'wiki_auto', 'wiki_bio', 'wiki_lingua',
        'wiki_source', 'wiki_summary', 'wmt20_mlqe_task1', 'wmt20_mlqe_task2',
        'wmt20_mlqe_task3', 'xglue', 'xsum_factuality'
    ],
    'structure-prediction': [
        'acronym_identification', 'ade_corpus_v2', 'afrikaans_ner_corpus',
        'alt', 'amttl', 'arabic_pos_dialect', 'bc2gm_corpus', 'best2009',
        'caner', 'coached_conv_pref', 'conll2002', 'conll2003', 'conllpp',
        'dane', 'ehealth_kd', 'EMBO/sd-nlp', 'euronews', 'finer',
        'german_legal_entity_recognition', 'germaner', 'harem', 'hausa_voa_ner',
        'id_nergrit_corpus', 'igbo_ner', 'indonlu', 'irc_disentangle',
        'isixhosa_ner_corpus', 'isizulu_ner_corpus', 'Jean-Baptiste/wikiner_fr',
        'jnlpba', 'kor_ner', 'lener_br', 'limit', 'linnaeus', 'lst20',
        'mac_morpho', 'msra_ner', 'multi_woz_v22', 'NbAiLab/norne',
        'ncbi_disease', 'nchlt', 'nkjp-ner', 'norec', 'norne', 'norwegian_ner',
        'numeric_fused_head', 'peoples_daily_ner', 'persian_ner', 'ronec',
        'schema_guided_dstc8', 'sem_eval_2020_task_11', 'senti_ws',
        'sepedi_ner', 'sesotho_ner_corpus', 'setswana_ner_corpus',
        'siswati_ner_corpus', 'smartdata', 'sofc_materials_articles',
        'species_800', 'swedish_ner_corpus', 'thainer', 'turkish_ner',
        'turkish_shrinked_ner', 'turku_ner_corpus', 'universal_morphologies',
        'weibo_ner', 'wikiann', 'wikicorpus', 'wino_bias', 'winograd_wsc',
        'wisesight1000', 'woz_dialogue', 'xglue', 'yoruba_gv_ner', 'zest'
    ],
    'question-answering': [
        'adversarial_qa', 'ambig_qa', 'aqua_rat', 'aquamuse', 'babi_qa', 'c3',
        'cbt', 'codah', 'covid_qa_castorini', 'covid_qa_deepset',
        'covid_qa_ucsd', 'cryptonite', 'doc2dial', 'dream', 'duorc', 'dyk',
        'dynabench/qa', 'eli5', 'exams', 'fquad', 'freebase_qa', 'grail_qa',
        'head_qa', 'hybrid_qa', 'iapp_wiki_qa_squad', 'indonlu', 'kilt_tasks',
        'lhoestq/custom_squad', 'liveqa', 'm_lama', 'mc_taco', 'med_hop',
        'medical_dialog', 'mkqa', 'mocha', 'mrqa', 'msr_sqa', 'multi_re_qa',
        'narrativeqa', 'narrativeqa_manual', 'neural_code_search', 'newsqa',
        'nq_open', 'parsinlu_reading_comprehension',
        'persiannlp/parsinlu_reading_comprehension', 'piqa', 'proto_qa',
        'pubmed_qa', 'qa_srl', 'qed', 'quac', 'reasoning_bg', 'ropes', 'selqa',
        'sharc', 'sharc_modified', 'simple_questions_v2', 'so_stacksample',
        'squad', 'squad_adversarial', 'squad_kor_v1', 'squad_kor_v2',
        'susumu2357/squad_v2_sv', 'thaiqa_squad', 'tweet_qa', 'wiki_hop',
        'wiki_movies', 'wiki_qa_ar', 'wiki_summary', 'xglue', 'xor_tydi_qa',
        'xquad_r', 'yahoo_answers_qa', 'zest'
    ],
    'sequence-modeling': [
        'air_dialogue', 'amazon_reviews_multi', 'arabic_billion_words', 'brwac',
        'bswac', 'cawac', 'cc_news', 'cc100', 'chr_en', 'coached_conv_pref',
        'code_search_net', 'craigslist_bargains', 'cs_restaurants',
        'curiosity_dialogs', 'dbrd', 'dialog_re', 'EMBO/biolang', 'farsi_news',
        'formermagic/github_python_1m', 'gem', 'glucose',
        'hebrew_projectbenyehuda', 'hebrew_this_world', 'hindi_discourse',
        'hkcancor', 'hrwac', 'id_newspapers_2018', 'id_puisi',
        'igbo_monolingual', 'irc_disentangle', 'kd_conv', 'kilt_tasks',
        'makhzan', 'mdd', 'meta_woz', 'miam', 'mkb', 'multi_woz_v22',
        'mutual_friends', 'newsph', 'numer_sense', 'opus100', 'oscar',
        'para_pat', 'pec', 'pib', 'ptb_text_only', 'pubmed', 'py_ast', 'quac',
        'recipe_nlg', 's2orc', 'sanskrit_classic', 'saudinewsnet',
        'schema_guided_dstc8', 'silicone', 'sofc_materials_articles',
        'spanish_billion_words', 'srwac', 'swahili', 'tashkeela', 'taskmaster1',
        'taskmaster2', 'taskmaster3', 'telugu_books', 'telugu_news', 'thaisum',
        'tlc', 'twi_text_c3', 'wikicorpus', 'wikitext_tl39', 'woz_dialogue',
        'yoruba_text_c3', 'youtube_caption_corrections'
    ],
    'other': [
        'aquamuse', 'ar_cov19', 'arabic_speech_corpus', 'cail2018', 'cbt',
        'ccaligned_multilingual', 'cdsc', 'cifar10', 'cifar100',
        'coached_conv_pref', 'common_voice', 'cord19', 'covost2',
        'crawl_domain', 'dialog_re', 'eth_py150_open', 'europarl_bilingual',
        'fashion_mnist', 'few_rel', 'generics_kb',
        'jimregan/clarinpl_sejmsenat', 'kelm', 'large_spanish_corpus',
        'librispeech_asr', 'lj_speech', 'medal', 'mnist', 'msr_genomics_kbcomp',
        'ollie', 'openslr', 're_dial', 's2orc', 'sent_comp',
        'spanish_billion_words', 'timit_asr', 'tuple_ie',
        'tweets_ar_en_parallel', 'universal_dependencies',
        'youtube_caption_corrections'
    ],
    'text-scoring': [
        'allegro_reviews', 'amazon_reviews_multi', 'app_reviews', 'asset',
        'assin', 'assin2', 'climate_fever', 'conv_ai', 'conv_ai_2', 'conv_ai_3',
        'counter', 'crows_pairs', 'google_wellformed_query', 'has_part',
        'hate_speech_pl', 'hippocorpus', 'humicroedit', 'kor_nlu', 'lama',
        'lavis-nlp/german_legal_sentences', 'm_lama', 'multi_nli', 'newspop',
        'oclar', 'paws', 'paws-x', 'pubmed', 'refresd', 'ro_sts',
        'sem_eval_2014_task_1', 'senti_ws', 'silicone', 'sst', 'stsb_mt_sv',
        'stsb_multi_mt', 'twi_wordsim353', 'yoruba_wordsim353'
    ],
    'text-retrieval': [
        'arxiv_dataset', 'bprec', 'climate_fever', 'eu_regulatory_ir',
        'evidence_infer_treatment', 'fquad', 'hover', 'kilt_tasks', 'lama',
        'lavis-nlp/german_legal_sentences', 'metooma', 'nell', 'pec',
        'recipe_nlg', 'times_of_india_news_headlines'
    ],
    'translation': [
        'persiannlp/parsinlu_translation_en_fa',
        'persiannlp/parsinlu_translation_fa_en'
    ],
    'cross-language-transcription': ['Yves/fhnw_swiss_parliament'],
    'textual-entailment': ['persiannlp/parsinlu_entailment'],
    'natural-language-inference': ['persiannlp/parsinlu_entailment'],
    'query-paraphrasing': ['persiannlp/parsinlu_query_paraphrasing'],
    'sentiment-analysis': ['persiannlp/parsinlu_sentiment']
}


def load_model(name: str) -> dict:
  return {'name': name, 'source': 'huggingface'}


def load_dataset(name: str) -> dict:
  return {'name': name, 'source': 'huggingface'}
