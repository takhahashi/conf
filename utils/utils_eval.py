from torch.utils.data import DataLoader
import numpy as np
from utils.utils_data import data_collator

def evaluate_model(config, model, datasets):
    test_dataloader = DataLoader(datasets['test'], batch_size=8, shuffle=False, collate_fn=data_collator)
    
    model.eval()
    hidden_states = []
    reg_output = []
    logits = []
    for step, inputs in enumerate(test_dataloader):
        print(inputs['input_ids'])
        print(inputs['token_type_ids'])
        print(inputs['attention_mask'])
        inputs = data_collator(inputs)
        bert_input = {'input_ids':inputs['input_ids'], 
                      'token_type_ids':inputs['token_type_ids'], 
                      'attention_mask':inputs['attention_mask']}
        print(inputs['input_ids'])
        print(inputs['token_type_ids'])
        print(inputs['attention_mask'])
        outputs = model(**bert_input, output_hidden_states=True)
        hidden_states.append(outputs.hidden_states[-1][:, 0, :].to('cpu').detach().numpy().copy())
        if config.model.model_type == "hybrid":
            reg_output.append(outputs.reg_output.to('cpu').detach().numpy().copy())
            logits.append(outputs.logits.to('cpu').detach().numpy().copy())
    reg_output = list(np.concatenate(reg_output))
    logits = list(np.concatenate(logits))
    hidden_states = list(np.concatenate(hidden_states))

    return {"reg_output":reg_output, "logits":logits, "hidden_states":hidden_states}