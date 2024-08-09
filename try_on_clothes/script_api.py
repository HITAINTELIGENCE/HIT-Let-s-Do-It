class Config:
    def __init__(self, checkpoint, data_root, out_dir, name, batch_size, n_worker, gpu_id, log_freq, radius, fine_width, fine_height, grid_size):
        self.checkpoint = checkpoint
        self.data_root = data_root
        self.out_dir = out_dir
        self.name = name
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.gpu_id = gpu_id
        self.log_freq = log_freq
        self.radius = radius
        self.fine_width = fine_width
        self.fine_height = fine_height
        self.grid_size = grid_size

def predict():
    from try_on_clothes.src.run_gmm import run, GMM, GMMDataset, DataLoader, torch
    from try_on_clothes.utils.utils import load_checkpoint, save_checkpoint

    # Khởi tạo Config và GMM
    config_gmm = Config(checkpoint='pre_trained/gmm_final.pth',
                        data_root='Database',
                        out_dir='output/first',
                        name='GMM',
                        batch_size=16,
                        n_worker=4,
                        gpu_id='0',
                        log_freq=100,
                        radius=5,
                        fine_width=192,
                        fine_height=256,
                        grid_size=5)
    model_gmm = GMM(config_gmm)
    load_checkpoint(model_gmm, config_gmm.checkpoint)
    model_gmm.eval()
    print('Run on {} data'.format("VAL"))
    dataset_gmm = GMMDataset(config_gmm, "val", data_list='val_pairs.txt', train=False)
    dataloader_gmm = DataLoader(dataset_gmm, batch_size=config_gmm.batch_size,
                                num_workers=config_gmm.n_worker, shuffle=False)
    with torch.no_grad():
        run(config_gmm, model_gmm, dataloader_gmm, "val")
    print('Successfully completed')

    # Khởi tạo Config và TOM
    from try_on_clothes.src.run_tom import run, UnetGenerator, nn, TOMDataset, DataLoader
    config_tom = Config(checkpoint='pre_trained/tom_final.pth',
                        data_root='Database',
                        out_dir='output/second',
                        name='TOM',
                        batch_size=16,
                        n_worker=4,
                        gpu_id='0',
                        log_freq=100,
                        radius=5,
                        fine_width=192,
                        fine_height=256,
                        grid_size=5)
    model_tom = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    load_checkpoint(model_tom, config_tom.checkpoint)
    model_tom.eval()
    mode = 'val'
    print('Run on {} data'.format(mode.upper()))
    dataset_tom = TOMDataset(config_tom, mode, data_list=mode+'_pairs.txt', train=False)
    dataloader_tom = DataLoader(dataset_tom, batch_size=config_tom.batch_size, num_workers=config_tom.n_worker, shuffle=False)
    with torch.no_grad():
        run(config_tom, model_tom, dataloader_tom, mode)
    print('Successfully completed')
