import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

from runners.trainer import FireTrainer
from data_pipeline.dataloader import FireDataset, FirePropagation, AugTransform
from models.Seq2Seq import Seq2Seq
from models.Regressor import DirectRegressor
from models.archive.BiMamba import BiMamba
#from models.archive.Fourier_NN import Net_Model
from loss.segmentation_loss import SegmentLoss
from evaluation.calculate_kpis import calculate_kpis
from evaluation.animate import create_animation, create_inference, invert_1d_boundary

def parse_args():
    parser = argparse.ArgumentParser(description='Train or infer')
    parser.add_argument('config', help='Training config path')
    parser.add_argument(
        '--train',
        default=False,
        action='store_true',
        help='bool to enable training'
    )
    parser.add_argument(
        '--eval',
        default=False,
        action='store_true',
        help='bool to enable evaluation'
    )
    parser.add_argument(
        '--logs',
        help='the dir to save logs and checkpoints',
        default=None,
    )
    parser.add_argument(
        '--checkpoint_file',
        help='checkpoint to train, infer',
        default=None,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()

    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.train:
        root = config["Root_directory"]
        config["Dataset_pipeline"]["data_csv"] = config["Dataset_pipeline"]["data_csv"].format(Root_directory=root)
        config["Dataset_pipeline"]["val_csv"] = config["Dataset_pipeline"]["val_csv"].format(Root_directory=root)
        config["Dataset_pipeline"]["data_dir"] = config["Dataset_pipeline"]["data_dir"].format(Root_directory=root)

        transform = AugTransform(spatial_size=config['Dataset_pipeline']['spatial_size'])
        base_dataset = FireDataset(data_cfg=config['Dataset_pipeline'], transform=transform)
        dataset = FirePropagation(base_dataset, config['Dataset_pipeline'] )
        
        print('Number of samples loaded for training: ', len(dataset))

        batch_size = config['Training_pipeline']['batch_size']
        num_epochs = config['Training_pipeline']['epochs']
        checkpoint_interval = config['Training_pipeline']['checkpoint_interval']
        workers = config['Training_pipeline']['workers']
        Enable_validation = config['Enable_validation']
        check_val = config['Training_pipeline']['check_val_every_n_epoch'] if Enable_validation else 1

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )

        if 'DirectRegressor' in config['Model_parameters']:
            model = DirectRegressor(**config['Model_parameters']['DirectRegressor'])
        elif 'Seq2Seq' in config['Model_parameters']:
            model = Seq2Seq(**config['Model_parameters']['Seq2Seq'])
        elif 'Fourier_NN' in config['Model_parameters']:
            model = Net_Model(**config['Model_parameters']['Fourier_NN'])
        elif 'BiMamba' in config['Model_parameters']:
            model = BiMamba(**config['Model_parameters']['BiMamba'])
        else:
            raise ValueError("No valid model found in configuration.")

        loss = SegmentLoss(**config['Loss_parameters'])
        if args.checkpoint_file:
            lightning_model = FireTrainer.load_from_checkpoint(
                    args.checkpoint_file, model=model, loss=loss,
                    optim_parameters=config['Training_pipeline']['optim_parameters'])
        else:
            lightning_model = FireTrainer(model, loss=loss,
                                optim_parameters=config['Training_pipeline']['optim_parameters'])


        logger = TensorBoardLogger(args.logs, name="training")

        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=args.logs,
            filename='model-{epoch:02d}',
            every_n_epochs = checkpoint_interval,
            save_top_k=num_epochs // checkpoint_interval,
        )

        lr_monitor_callback = LearningRateMonitor(logging_interval='step')

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            logger=logger,
            callbacks=[
                checkpoint_callback,
                lr_monitor_callback
            ],
            accelerator='auto',
            check_val_every_n_epoch = check_val,
            num_sanity_val_steps=0,
            #profiler="advanced"
        )
        if Enable_validation:
            base_dataset = FireDataset(data_cfg=config['Dataset_pipeline'], transform=None,
                                       Enable_validation=True)
            dataset = FirePropagation(base_dataset, config['Dataset_pipeline'])
            val_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
            )
        
            trainer.fit(lightning_model, train_dataloaders=dataloader, val_dataloaders=val_loader)
            #trainer.validate(model=lightning_model)
        else:
            trainer.fit(lightning_model, train_dataloaders=dataloader)
    
    elif args.eval:
        root = config["Root_directory"]
        config["Dataset_pipeline"]["data_csv"] = config["Dataset_pipeline"]["data_csv"].format(Root_directory=root)
        config["Dataset_pipeline"]["val_csv"] = config["Dataset_pipeline"]["val_csv"].format(Root_directory=root)
        config["Dataset_pipeline"]["data_dir"] = config["Dataset_pipeline"]["data_dir"].format(Root_directory=root)
        
        batch_size = config['Eval_pipeline']['batch_size']
        workers = config['Eval_pipeline']['workers']

        base_dataset = FireDataset(data_cfg=config['Dataset_pipeline'], transform=None,
                                       Enable_validation=config['Enable_validation'])
        dataset = FirePropagation(base_dataset, config['Dataset_pipeline'])
        test_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
            )

        if 'DirectRegressor' in config['Model_parameters']:
            model = DirectRegressor(**config['Model_parameters']['DirectRegressor'])
        elif 'Seq2Seq' in config['Model_parameters']:
            model = Seq2Seq(**config['Model_parameters']['Seq2Seq'])
        elif 'Fourier_NN' in config['Model_parameters']:
            model = Net_Model(**config['Model_parameters']['Fourier_NN'])
        elif 'BiMamba' in config['Model_parameters']:
            model = BiMamba(**config['Model_parameters']['BiMamba'])
        else:
            raise ValueError("No valid model found in configuration.")

        lightning_model = FireTrainer.load_from_checkpoint(
                    args.checkpoint_file, model=model)

        lightning_model.eval()
        if config['Eval_pipeline']['mode'] == 'Inference':
            test_loader = DataLoader(
                dataset=base_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
            )
            image_shape = (config["Dataset_pipeline"]["out_frames"],*config["Dataset_pipeline"]["spatial_size"])
            data = np.genfromtxt(config["Dataset_pipeline"]["data_csv"], delimiter=',', dtype=None, names=True, encoding='utf-8')
            ids = data['id']
            collect_results = {}
            with torch.no_grad():
                for i, x in enumerate(test_loader):
                    x = x.to(device)
                    probs = lightning_model(x)
                    collect_results[ids[i]] = probs.cpu().numpy().flatten(order='C').astype(np.float32)

            df = pd.DataFrame.from_dict(collect_results, orient='index')
            df['id'] = ids
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df = df[cols]
            df = df.reset_index(drop=True)
            df.to_csv(args.logs + 'submission.csv',index=False)
            print(f"Results saved to {args.logs + 'submission.csv'}")
            create_inference(collect_results, data, image_shape, folder=args.logs, threshold=None)


        elif config['Eval_pipeline']['mode'] == 'Regular':
            collect_results = []
            collect_targets = []
            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    probs = lightning_model(x)
                    #probs = invert_1d_boundary(probs, H=y.shape[-2]).unsqueeze(2)
                    collect_results.append(probs)
                    collect_targets.append(y)

            kpis = calculate_kpis(torch.concat(collect_results, axis=0), 
                                torch.concat(collect_targets, axis=0))
            print(kpis)
            create_animation(torch.concat(collect_results, axis=0), torch.concat(collect_targets, axis=0), 
                            output_file='animation.mp4', threshold=0.8)
        ### Autoregressive evaluation
        elif config['Eval_pipeline']['mode'] == 'Autoregressive':
            collect_results = []
            collect_targets = []
            x = test_loader.dataset[0][0].unsqueeze(0)
            static = x[:,:,3:].to(device)
            in_frames = config['Dataset_pipeline']['in_frames']
            out_frames = config['Dataset_pipeline']['out_frames']
            with torch.no_grad():
                for batch in test_loader:
                    _, y = batch
                    x, y = x.to(device), y.to(device)
                    preceeding_frames = x[:, out_frames:in_frames, :3] if in_frames > out_frames else x[:, :in_frames]
                    x = lightning_model(x)
                    collect_results.append(x)
                    collect_targets.append(y)
                    if in_frames < out_frames:
                        x = torch.cat([x[:, out_frames-in_frames:out_frames], static], axis=2)
                    else:
                        x = torch.cat([preceeding_frames, x], axis=1)
                        x = torch.cat([x, static], axis=2)

            
            kpis = calculate_kpis(torch.concat(collect_results, axis=1), 
                                torch.concat(collect_targets, axis=1))

            create_animation(torch.concat(collect_results, axis=1), torch.concat(collect_targets, axis=1), 
                            output_file='animation.mp4', threshold=0.8)
            print(kpis)
        else:
            raise ValueError("No valid evaluation mode found in config.")
    