def bertfncall():
    model_class = transformers.BertModel
    tokenizer_class = transformers.BertTokenizer
    pretrained_weights='bert-base-uncased'

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    bert_model = model_class.from_pretrained(pretrained_weights)

    #converting comments into a 2d matrix

    max_seq = 100

    def tokenize_text(df, max_seq):
        return [
            tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in df.comment_text.values
        ]


    def pad_text(tokenized_text, max_seq):
        return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])


    def tokenize_and_pad_text(df, max_seq):
        tokenized_text = tokenize_text(df, max_seq)
        padded_text = pad_text(tokenized_text, max_seq)
        return torch.tensor(padded_text)


    def targets_to_tensor(df, target_columns):
        return torch.tensor(df[target_columns].values, dtype=torch.float32)

    #tokenize, pad and convert comments to PyTorch Tensors. Then we use BERT to transform the text to embeddings

    train_indices = tokenize_and_pad_text(df_train, max_seq)
    val_indices = tokenize_and_pad_text(df_val, max_seq)
    test_indices = tokenize_and_pad_text(df_test, max_seq)

    with torch.no_grad():
        x_train = bert_model(train_indices)[0]  
        x_val = bert_model(val_indices)[0]
        x_test = bert_model(test_indices)[0]

    y_train = targets_to_tensor(df_train, target_columns)
    y_val = targets_to_tensor(df_val, target_columns)
    y_test = targets_to_tensor(df_test, target_columns)

    #This is the first comment transformed into word embeddings with BERT. It has a [100 x 768] shape
    x_train[0]
    y_train[0]
