import random
from collections import OrderedDict
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoTokenizer,
    DataCollatorWithPadding,
)


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")

def load_data():
    """Load IMDB data (training and eval)"""
    seed = random.randint(0, 100)
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=seed)

    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")

    # random 10 samples
    population = random.sample(range(len(raw_datasets["train"])), 10)

    tokenized_datasets = raw_datasets.map(
        lambda examples: tokenizer(examples["text"], truncation=True), batched=True
    )
    tokenized_datasets["train"] = tokenized_datasets["train"].select(population)
    tokenized_datasets["test"] = tokenized_datasets["test"].select(population)

    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def train(net, trainloader, epochs ,learning_rate):
    optimizer = AdamW(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy



