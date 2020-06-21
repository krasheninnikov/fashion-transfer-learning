import numpy as np
import torch
import matplotlib.pyplot as plt


def get_class_weights(styles_df_train, sorted_class_names):
    """
    Classes are downweighted proportionally to their frequency in the train set to ameliorate the class imbalance.
    If the class is not present in the train set, it gets the weighted as if there is one sample of that class.
    """
    train_class_counts = styles_df_train.groupby(['articleType']).size().to_dict()
    class_weights = np.ones(len(sorted_class_names))
    for i, c in enumerate(sorted_class_names):
        if c in train_class_counts:
            class_weights[i] = train_class_counts[c]
    class_weights = np.sum(class_weights) / class_weights
    class_weights = class_weights / np.sum(class_weights)
    return class_weights


def plot_losses(train_losses, val_losses):
    plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')
    plt.plot(np.arange(len(val_losses)), val_losses, label='val loss')
    plt.legend()
    plt.xlabel('#epochs')
    plt.ylabel('loss')
    plt.show()


def correct_top_k(dataloader, model, k_list, n_classes, device):
    """Computes per-class top-k correct counts for the values of k provided in k_list"""
    maxk = max(k_list)
    class_counts = np.zeros(n_classes)
    class_correct_topk = {k: np.zeros(n_classes) for k in k_list}
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, topk_pred = outputs.topk(maxk, 1, True, True)
            topk_pred = topk_pred.t()
            topk_correct = topk_pred.eq(labels.view(1, -1).expand_as(topk_pred)).cpu().numpy().T
            labels = labels.cpu().numpy()
            for i, label in enumerate(labels):
                class_counts[label] += 1
                for k in k_list:
                    class_correct_topk[k][label] += np.sum(topk_correct[i, :k])
            # print({k: class_correct_topk[k] / class_counts for k in k_list})
    return class_correct_topk, class_counts


def get_accuracy(correct_top_k_per_class, class_counts):
    per_class_accuracy = {}
    for k in correct_top_k_per_class:
        print('Top-{} accuracy:'.format(k))
        per_class_accuracy[k] = np.divide(correct_top_k_per_class[k],
                                          class_counts,
                                          out=np.zeros_like(class_counts),
                                          where=class_counts != 0)

        accuracy_average = np.sum(correct_top_k_per_class[k]) / np.sum(class_counts)
        print('Accuracy on the entire test set: {} ({}/{})'.format(accuracy_average,
                                                                   int(np.sum(correct_top_k_per_class[k])),
                                                                   int(np.sum(class_counts))))
        accuracy_top20 = np.sum(correct_top_k_per_class[k][:20]) / np.sum(class_counts[:20])
        print('Accuracy on 20 most frequent classes: {} ({}/{})'.format(accuracy_top20,
                                                                        int(np.sum(correct_top_k_per_class[k][:20])),
                                                                        int(np.sum(class_counts[:20]))))
    return per_class_accuracy


def compute_loss(dataloader, model, criterion, device):
    model = model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss


def train(dataloader_train,
          dataloader_val,
          n_epochs,
          model,
          criterion,
          optimizer,
          device,
          load_checkpoint_path=None,
          checkpoint_save_path=None):
    if load_checkpoint_path:
        checkpoint = torch.load(load_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    train_losses, val_losses = [], []
    prev_min_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = 0.0
        model = model.train()
        for i, (inputs, labels) in enumerate(dataloader_train):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # normalize the losses
        train_loss = train_loss / len(dataloader_train.dataset)
        val_loss = compute_loss(dataloader_val, model, criterion, device) / len(dataloader_val.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print('Epoch {}: train loss {}, val loss {}'.format(epoch + 1, train_loss, val_loss))

        # save the model and the training params if val loss is at its lowest so far
        if val_loss < prev_min_val_loss:
            prev_min_val_loss = val_loss
            if checkpoint_save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, checkpoint_save_path)
    return train_losses, val_losses, model
