def trainerfncall():
    class KimCNN(nn.Module):
        def __init__(self, embed_num, embed_dim, class_num, kernel_num, kernel_sizes, dropout, static):
            super(KimCNN, self).__init__()

            V = embed_num
            D = embed_dim
            C = class_num
            Co = kernel_num
            Ks = kernel_sizes

            self.static = static
            self.embed = nn.Embedding(V, D)
            self.convs1 = nn.ModuleList([nn.Conv2d(1, Co, (K, D)) for K in Ks])
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(len(Ks) * Co, C)
            self.sigmoid = nn.Sigmoid()


        def forward(self, x):
            if self.static:
                x = Variable(x)

            x = x.unsqueeze(1)  # (N, Ci, W, D)

            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

            x = torch.cat(x, 1)
            x = self.dropout(x)  # (N, len(Ks)*Co)
            logit = self.fc1(x)  # (N, C)
            output = self.sigmoid(logit)
            return output

    embed_num = x_train.shape[1]
    embed_dim = x_train.shape[2]
    class_num = y_train.shape[1]
    kernel_num = 3
    kernel_sizes = [2, 3, 4]
    dropout = 0.5
    static = True
    model = KimCNN(
        embed_num=embed_num,
        embed_dim=embed_dim,
        class_num=class_num,
        kernel_num=kernel_num,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
        static=static,
    )


    #training for10 epochs
    n_epochs = 10
    batch_size = 10
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    #generate training data
    def generate_batch_data(x, y, batch_size):
        i, batch = 0, 0
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size), 1):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch
        if i + batch_size < len(x):
            yield x[i + batch_size :], y[i + batch_size :], batch + 1
        if batch == 0:
            yield x, y, 1

     train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0

        model.train(True)
        for x_batch, y_batch, batch in generate_batch_data(x_train, y_train, batch_size):
            y_pred = model(x_batch)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0

        model.train(True)
        for x_batch, y_batch, batch in generate_batch_data(x_train, y_train, batch_size):
            y_pred = model(x_batch)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= batch
        train_losses.append(train_loss)
        elapsed = time.time() - start_time

        model.eval() # disable dropout for deterministic output
        with torch.no_grad(): # deactivate autograd engine to reduce memory usage and speed up computations
            val_loss, batch = 0, 1
            for x_batch, y_batch, batch in generate_batch_data(x_val, y_val, batch_size):
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                val_loss += loss.item()
            val_loss /= batch
            val_losses.append(val_loss)

        print(
            "Epoch %d Train loss: %.2f. Validation loss: %.2f. Elapsed time: %.2fs."
            % (epoch + 1, train_losses[-1], val_losses[-1], elapsed)
        )

       #observation
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")

    #testing
    model.eval() # disable dropout for deterministic output
    with torch.no_grad(): # deactivate autograd engine to reduce memory usage and speed up computations
        y_preds = []
        batch = 0
        for x_batch, y_batch, batch in generate_batch_data(x_test, y_test, batch_size):
            y_pred = model(x_batch)
            y_preds.extend(y_pred.cpu().numpy().tolist())
        y_preds_np = np.array(y_preds)
    y_preds_np

    y_test_np = df_test[target_columns].values
    y_test_np[1000:]

    #extract real labels of toxicity threats for the test set
    y_test_np = df_test[target_columns].values
    y_test_np[1000:]

    auc_scores = roc_auc_score(y_test_np, y_preds_np, average=None)
    df_accuracy = pd.DataFrame({"label": target_columns, "auc": auc_scores})
    df_accuracy.sort_values('auc')[::-1]


    #checking if we have an imbalanced dataset
    positive_labels = df_train[target_columns].sum().sum()
    positive_labels
    # Output:
    2201
    all_labels = df_train[target_columns].count().sum()
    all_labels
    # Output:
    60000
    positive_labels/all_labels
    # Output:
    0.03668333333333333


    #sanity check

    df_test_targets = df_test[target_columns]
    df_pred_targets = pd.DataFrame(y_preds_np.round(), columns=target_columns, dtype=int)
    df_sanity = df_test_targets.join(df_pred_targets, how='inner', rsuffix='_pred')
    df_test_targets.sum()
    df_pred_targets.sum()
    df_sanity[df_sanity.toxic > 0][['toxic', 'toxic_pred']]
