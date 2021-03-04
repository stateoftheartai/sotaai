# Hugging Face
 
 Wrapper `hugggingface_wrapper.py`

 General notes:

 - Wraps datasets and models of the [Hugging Face](https://huggingface.co) library
 - Provides standarized functions to load datasets and models: `load_dataset` and `load_model`

 ## Datasets

 Function: `load_dataset`

 Return type: [`datasets.Dataset`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#dataset) or [`datasets.DatasetDict`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasetdict).

if `split` is not None: the dataset requested,
if `split` is None, a `datasets.DatasetDict` with each split.

 ## Models

 Function: `load_model`

 Return Type: Any of the following:

- wav2vec2 – `Wav2Vec2Model` (Wav2Vec2 model)
  
- convbert – `ConvBertModel` (ConvBERT model)
  
- led – `LEDModel` (LED model)
  
- blenderbot-small – `BlenderbotSmallModel` (BlenderbotSmall model)
  
- retribert – `RetriBertModel` (RetriBERT model)
  
- mt5 – `MT5Model` (mT5 model)
  
- t5 – `T5Model` (T5 model)
  
- mobilebert – `MobileBertModel` (MobileBERT model)
  
- distilbert – `DistilBertModel` (DistilBERT model)
  
- albert – `AlbertModel` (ALBERT model)
  
- bert-generation – `BertGenerationEncoder` (Bert Generation model)
  
- camembert – `CamembertModel` (CamemBERT model)
  
- xlm-roberta – `XLMRobertaModel` (XLM-RoBERTa model)
  
- pegasus – `PegasusModel` (Pegasus model)
  
- marian – `MarianModel` (Marian model)
  
- mbart – `MBartModel` (mBART model)
  
- mpnet – `MPNetModel` (MPNet model)
  
- bart – `BartModel` (BART model)
  
- blenderbot – `BlenderbotModel` (Blenderbot model)
  
- reformer – `ReformerModel` (Reformer model)
  
- longformer – `LongformerModel` (Longformer model)
  
- roberta – `RobertaModel` (RoBERTa model)
  
- deberta – `DebertaModel` (DeBERTa model)
  
- flaubert – `FlaubertModel` (FlauBERT model)
  
- fsmt – `FSMTModel` (FairSeq Machine-Translation model)
  
- squeezebert – `SqueezeBertModel` (SqueezeBERT model)
  
- bert – `BertModel` (BERT model)
  
- openai-gpt – `OpenAIGPTModel` (OpenAI GPT model)
  
- gpt2 – `GPT2Model` (OpenAI GPT-2 model)
  
- transfo-xl – `TransfoXLModel` (Transformer-XL model)
  
- xlnet – `XLNetModel` (XLNet model)
  
- xlm-prophetnet – `XLMProphetNetModel` (XLMProphetNet model)
  
- prophetnet – `ProphetNetModel` (ProphetNet model)
  
- xlm – `XLMModel` (XLM model)
  
- ctrl – `CTRLModel` (CTRL model)
  
- electra – `ElectraModel` (ELECTRA model)
  
- funnel – `FunnelModel` (Funnel Transformer model)
  
- lxmert – `LxmertModel` (LXMERT model)
  
- dpr – `DPRQuestionEncoder` (DPR model)
  
- layoutlm – `LayoutLMModel` (LayoutLM model)
  
- tapas – `TapasModel` (TAPAS model)
  
Notes:

- All models inherit from [`PreTrainedModel`](https://huggingface.co/transformers/main_classes/model.html#pretrainedmodel), then `PreTrainedModel` inherirts from [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)
