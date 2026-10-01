import torch

from models.inversion_net import InversionNet_modified


def train_inversion_net(data, epochs, device, model=None, optimizer=None, learning_rate=0.0001):
    model = InversionNet_modified().to(device) if model is None else model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) if optimizer is None else optimizer
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for observation, sound_speed in data:
            observation = observation.to(device)
            sound_speed = sound_speed.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(observation)
            loss = loss_fn(prediction, sound_speed)
            loss.backward()
            optimizer.step()
    return model
