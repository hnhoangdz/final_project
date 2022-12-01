import os
import torch

# Save weights
def save(net, logger, model_save_dir, exp="best"):
    # path to save model according best or last
    path = os.path.join(model_save_dir, exp)

    # checkpoint
    checkpoint = {
        'logs': logger.get_logs(),
        'params': net.state_dict()
    }

    # save checkpoint
    torch.save(checkpoint, path + ".pth")

# Load trained weights
def restore(net, logger, hps, optimizer, exp="best"):
    """ Load back the model and logger from a given checkpoint
        epoch detailed in hps['restore_epoch'], if available"""
        
    path = os.path.join(hps['model_save_dir'] + \
                            "_bs_" + str(hps['batch_size']) + \
                            "_" + optimizer + "_" + \
                            "_lr_" + str(hps['lr']), exp)

    if os.path.exists(path):
        try:
            checkpoint = torch.load(path)
            logger.restore_logs(checkpoint['logs'])
            net.load_state_dict(checkpoint['params'])
            print("Network Restored!")

        except Exception as e:
            print("Restore Failed! Training from scratch.")
            print(e)
            hps['start_epoch'] = 0

    else:
        print("Restore point unavailable. Training from scratch.")
        hps['start_epoch'] = 0

def load_features(model, params):
    """ Load params into all layers of 'model'
        that are compatible, then freeze them"""
        
    model_dict = model.state_dict()

    imp_params = {k: v for k, v in params.items() if k in model_dict}

    # Load layers
    model_dict.update(imp_params)
    model.load_state_dict(imp_params)

    # Freeze layers
    for name, param in model.named_parameters():
        param.requires_grad = False
