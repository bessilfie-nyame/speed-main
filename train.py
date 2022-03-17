import torch
from torch import import_ir_module
from ignite.engine import Engine, _prepare_batch, create_supervised_trainer, Events, create_supervised_evaluator
from ignite.metrics import MeanSquaredError, MeanAbsoluteError
from torch.optim.lr_scheduler import ExponentialLR
from ignite.handlers import ModelCheckpoint, Checkpoint, EarlyStopping
import numpy as np
from torch.utils.data.dataset import Subset

import json
import os

from loader import loaders, train_dataset, val_dataset
from models.model_net import I3DR50


with open(os.path.join(os.getcwd(), 'config.json'), "r") as config_file:
    config = json.load(config_file)

model = I3DR50(num_classes=1, init_model='resnet')

device = "cuda"
if not torch.cuda.is_available():
    device = "cpu"

else:
    model = model.to(torch.device("cuda"))

print("PyTorch version: {} | Device: {}".format(torch.__version__, device))
print("Train loader: num_batches={} | num_samples={}".format(len(loaders["train"]), len(loaders["train"].sampler)))
print("Validation loader: num_batches={} | num_samples={}".format(len(loaders["val"]), len(loaders["val"].sampler)))

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.05)
criterion = torch.nn.functional.mse_loss

trainer = create_supervised_trainer(model, optimizer, criterion, device)

# Loss logging
log_interval = 50 
if 'cpu' in device:
    log_interval = 5 

@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iteration = (engine.state.iteration - 1) % len(loaders["train"]) + 1
    if iteration % log_interval == 0:
        print("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(
            engine.state.epoch, iteration, len(loaders["train"]), engine.state.output))

# Metrics

metrics = {
    'avg_mse': MeanSquaredError(),
    'avg_mae': MeanAbsoluteError()
}

train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)

random_indices = np.random.permutation(np.arange(len(train_dataset)))[:len(val_dataset)]
train_subset = Subset(train_dataset, indices=random_indices)

train_eval_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=4, 
                               drop_last=True, pin_memory="cuda" in device)

@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_offline_train_metrics(engine):
    epoch = engine.state.epoch
    print("Compute train metrics...")
    metrics = train_evaluator.run(train_eval_loader).metrics
    print("Training Results - Epoch: {}  Average MSE Loss: {:.4f} Average MAE: {:.4f}"
          .format(engine.state.epoch, metrics['avg_mse'], metrics['avg_mae']))    
    
@trainer.on(Events.EPOCH_COMPLETED)
def compute_and_display_val_metrics(engine):
    epoch = engine.state.epoch
    print("Compute validation metrics...")
    metrics = val_evaluator.run(loaders["val"]).metrics
    print("Validation Results - Epoch: {}  Average MSE Loss: {:.4f} Average MAE: {:.4f}"
          .format(engine.state.epoch, metrics['avg_mse'], metrics['avg_mae']))

# Learning rate scheduling
# lr_scheduler = ExponentialLR(optimizer, gamma=0.8)

# @trainer.on(Events.EPOCH_STARTED)
# def update_lr_scheduler(engine):
#     lr_scheduler.step()
#     # Display learning rate:
#     if len(optimizer.param_groups) == 1:
#         lr = float(optimizer.param_groups[0]['lr'])
#         print("Learning rate: {}".format(lr))
#     else:
#         for i, param_group in enumerate(optimizer.param_groups):
#             lr = float(param_group['lr'])
#             print("Learning rate (group {}): {}".format(i, lr))

def score_function(engine):
    val_avg_mse = engine.state.metrics['avg_mse']
    # Objects with highest scores will be retained.
    return val_avg_mse

# Training checkpointing
# best_model_saver = ModelCheckpoint(os.path.join(os.getcwd(), config["best_models_path"]),  # folder where to save the best model(s)
#                                    filename_prefix="model",  # filename prefix -> {filename_prefix}_{name}_{step_number}_{score_name}={abs(score_function_result)}.pth
#                                    score_name="val_mse",  
#                                    score_function=score_function,
#                                    n_saved=3,
#                                    atomic=True,  # objects are saved to a temporary file and then moved to final destination, so that files are guaranteed to not be damaged
#                                    save_as_state_dict=True,  # Save object as state_dict
#                                    create_dir=True)

# val_evaluator.add_event_handler(Events.COMPLETED, best_model_saver, {"best_model": model})

# training_saver = ModelCheckpoint(os.path.join(os.getcwd(), config["checkpoint_path"]),
#                                  filename_prefix="checkpoint",
#                                  save_interval=None,
#                                  n_saved=3,
#                                  atomic=True,
#                                  save_as_state_dict=True,
#                                  create_dir=True)

# to_save = {"trainer": trainer, "model": model, "optimizer": optimizer} # "lr_scheduler": lr_scheduler
# trainer.add_event_handler(Events.ITERATION_COMPLETED, training_saver, to_save) 

# Early Stopping 
early_stopping = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)

val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)

# Train
max_epochs = 2

trainer.run(loaders["train"], max_epochs=max_epochs)



