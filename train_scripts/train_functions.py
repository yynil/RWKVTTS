def train_step(model,batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    output = model(batch)
    return output