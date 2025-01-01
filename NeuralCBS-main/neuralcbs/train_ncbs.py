import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Neural_CBS,LpLoss
from helm_datasets import datasetFactory
import matplotlib.pyplot as plt
import os
import logging

def main(args):
    logging.basicConfig(filename=args.log_path,
                        level=logging.INFO,
                        format='%(asctime)s-%(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device}")
    logging.info(f"Params: {vars(args)}")
    model = Neural_CBS(lefting=[2,10,args.lefting],
                    projection=[args.lefting,10,2],
                    padding=args.padding,
                    spatial_size=args.spatial_size).to(device)

    criterion = LpLoss()
    batchsize = args.batchsize
    train_loader, val_loader = datasetFactory(do='train',
                                                train_num=args.train_num,
                                                valid_num=args.val_num,
                                                batchsize=batchsize,
                                                PATH_data=args.datapath,
                                                PATH_target=args.datapath,
                                                PATH_src=args.srcpath,
                                                workers=args.workers,
                                                num_folder=(args.train_num+args.val_num)//900+1)

    num_steps = args.num_iters
    num_epoch = args.num_epoch

    train_losses = []
    val_losses = []
    for step in range(1,num_steps+1):
        # print(f'Start training the first {step} steps...')
        step_train_losses = []
        step_val_losses = []
        optimizer = optim.AdamW(model.parameters(),lr=args.lr)
        shedular = optim.lr_scheduler.OneCycleLR(optimizer,
                                                 max_lr=args.lr,
                                                 epochs=num_epoch,
                                                 steps_per_epoch=len(train_loader),
                                                 pct_start=0.05,
                                                 div_factor=100,
                                                 final_div_factor=1e4
                                                 )
        val_loss = 0.0
        for epoch in range(num_epoch):
            
            model.train()
            for batch_idx,data in enumerate(train_loader):
                sos,src,label = data
                sos = sos.to(device)
                src = src.to(device)
                psi = src.clone()# initial value: u_homo
                label = label.to(device)
                optimizer.zero_grad()
                psi = model.multi_step(psi,sos,src,step)
                loss = criterion(psi, label)
                loss.backward()
                optimizer.step()
                shedular.step()
                step_train_losses.append(loss.item())
                if batch_idx % 50 == 0:
                    logging.info(f'Steps: [{step:<3}/{num_steps}], Epoch: [{epoch:<3}/{num_epoch}], Batch [{batch_idx:<3}/{len(train_loader)}], Loss: {loss.item():.4e}, Val_Loss: {val_loss:.4e}, Lr: {shedular.get_last_lr()[0]:.4e}')

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    sos, src, label = val_data
                    psi = src.clone().to(device)
                    sos = sos.to(device)
                    src = src.to(device)
                    label = label.to(device)
                    psi = model.multi_step(psi, sos, src, step)
                    loss = criterion(psi, label)
                    val_loss += loss.item()
                val_loss /= len(val_loader)
                step_val_losses.append(val_loss)
        if args.plot_loss:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            plt.figure() 
            plt.plot(torch.arange(len(step_train_losses)),step_train_losses,label='Train loss')
            plt.plot(torch.linspace(0,len(step_train_losses),len(step_val_losses)),step_val_losses,label='Val loss')
            plt.xticks(torch.linspace(0, len(step_train_losses), len(step_val_losses)))
            plt.ylabel("Loss")
            plt.xlabel("Step")
            plt.title(f'Train log on step {step}')
            plt.legend()
            plt.savefig(os.path.join(args.save_path,f'step_{step}_train_loss.png'), bbox_inches='tight')
        train_losses.append(step_train_losses)
        val_losses.append(step_val_losses)


    if args.save_info:
        checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_steps':num_steps
        }
        
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(checkpoint, os.path.join(args.save_path,'checkpoint.pth'))

        logging.info(f"Model and losses saved in {args.save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=5, help='number of iterations to trian')
    parser.add_argument("--num_epoch", type=int, default=15, help='number of epoch for each iteration')
    parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
    parser.add_argument("--lefting", type=int, default=20, help='lefting channel')
    parser.add_argument("--padding", type=int, default=8, help='padding size')
    parser.add_argument("--spatial_size", type=int, default=480, help='spatial size')
    parser.add_argument("--batchsize", type=int, default=8, help='batchsize')
    parser.add_argument("--train_num", type=int, default=3000, help='train number of speed data')
    parser.add_argument("--val_num", type=int, default=100, help='validation number of speed data')
    parser.add_argument("--datapath", type=str, default='/work/helm_data', help='root path of data')
    parser.add_argument("--srcpath", type=str, default='/work/helm_data/u_homo/u_homo.npy', help='specific path of source ')
    parser.add_argument("--workers", type=int, default=20, help='num of workers to load data')
    parser.add_argument("--plot_loss", action='store_true', help='whether plot loss')
    parser.add_argument("--save_info", action='store_true', help='whether save model and losses')
    parser.add_argument("--log_path", type=str, default='./training.log', help='logging path')
    parser.add_argument("--save_path", type=str, default='./weighits', help='saving path')
    args = parser.parse_args()
    main(args)
