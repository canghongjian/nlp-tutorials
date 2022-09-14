# 2.1 conditonal_generation
# https://github.com/thunlp/OpenPrompt/blob/main/tutorial/2.1_conditional_generation.py
from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor
dataset = {}
dataset['train'] = WebNLGProcessor().get_train_examples("/root/datasets/CondGen/webnlg_2017/")
dataset['validation'] = WebNLGProcessor().get_dev_examples("/root/datasets/CondGen/webnlg_2017/")
dataset['test'] = WebNLGProcessor().get_test_examples("/root/datasets/CondGen/webnlg_2017/")

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm('t5', 't5-base')

# 使用prefix_tuning_template
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# 取一个训练数据作为样例观察template如何wrap
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)

from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=5,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=5,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=False)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from transformers import AdamW
# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]


optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)

from transformers.optimization import get_linear_schedule_with_warmup
import torch

tot_step  = len(train_dataloader)*5
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
    return generated_sentence




generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": [[628], [198]]
}

# training and generation.
global_step = 0
tot_loss = 0
log_loss = 0
for epoch in range(5):
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        global_step +=1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step %50 ==0:
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

generated_sentence = evaluate(prompt_model, test_dataloader)
with open(f"Generated_sentence_webnlg_gpt2_False.txt",'w') as f:
    for i in generated_sentence:
        f.write(i+"\n")
