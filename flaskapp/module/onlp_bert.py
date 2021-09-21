import torch
import numpy as np
from transformers import BertTokenizer#version 4.0.1
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

def load_pretrained():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    model = BertForSequenceClassification.from_pretrained(
                                                        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                                                        num_labels = 3, # The number of output labels--3 for sentiment classification.  
                                                        output_attentions = False, # Whether the model returns attentions weights.
                                                        output_hidden_states = False, # Whether the model returns all hidden-states.
                                                    )
    #load the pretrained model
    checkpoint = torch.load("data/model_2.pt",\
        map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model,tokenizer
    
def predict_sentiment(model,tokenizer,sent):
    #model.eval()
    input_ids = []
    attention_masks = []
    batch_size = 1
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        padding=True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )

    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])

    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    prediction_data = TensorDataset(input_ids, attention_masks,torch.tensor([0]))
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    for batch in prediction_dataloader:
        batch = tuple(t for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

        logits = outputs.logits

        # Move logits and labels to CPU
        logits = logits.numpy()
        label_ids = b_labels.numpy()
        
        # Store predictions and true labels
        pred_labels_i = np.argmax(logits, axis=1).flatten()
        prob = np.exp(logits)/(np.exp(logits).sum())

        return pred_labels_i[0],prob[0]

def trans_to_sentiment(prediction,prob):
    l = {2:"netural", 1:"positive",0:"negative"}
    label = l[prediction]
    prob = list(zip(l.values(),list(prob)))
    return label,prob[2],prob[1],prob[0]

if __name__ == "__main__":
    model,tokenizer = load_pretrained()
    sent = "covid is good in some way"
    prediction,prob = predict_sentiment(model,tokenizer,sent)
    l = {2:"netural", 1:"positive",0:"negative"}
    print('predicted as',str(l[prediction]))
    
    print(list(zip(l.values(),list(prob))))

