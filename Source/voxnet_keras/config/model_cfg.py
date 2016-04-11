class_id_to_name = {
    "1": "bathtub",
    "2": "bed",
    "3": "chair",
    "4": "desk",
    "5": "dresser",
    "6": "monitor",
    "7": "night_stand",
    "8": "sofa",
    "9": "table",
    "10": "toilet"
}
class_name_to_id = { v : k for k, v in class_id_to_name.items() }
class_names = set(class_id_to_name.values())

lr_schedule = { 0: 0.001,
                60000: 0.0001,
                400000: 0.00005,
                600000: 0.00001,
                }

voxnet_cfg = {'batch_size' : 32,
       'learning_rate' : lr_schedule,
       'reg' : 0.001,
       'momentum' : 0.9,
       'dims' : (32, 32, 32),
       'n_channels' : 1,
       'n_classes' : 10,
       'batches_per_chunk': 64,
       'max_epochs' : 80,
       'max_jitter_ij' : 2,
       'max_jitter_k' : 2,
       'n_rotations' : 12,
       'checkpoint_every_nth' : 4000,
       }