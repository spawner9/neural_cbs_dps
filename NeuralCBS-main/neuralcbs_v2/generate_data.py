import torch
from cbsmodel import ConvergentBornSeries



if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    model = ConvergentBornSeries(lamb=1,
                                sos=None,# default homogenous
                                dx=None,
                                src_loc=[100,40],
                                boundary_width=[8,8],
                                boundary_strength=1,
                                boundary_type='PML3').to(device)
    # generate training data
    num_train = 100
    data_train = []
    src_loces = torch.randint(20,108,(num_train,2))
    for i in range(num_train):
        model.set_src_loc(src_loc=src_loces[i])
        data_train.append(model(max_iters=5000,requries_grad=False).detach().cpu())
        print(f"processing {i}")
    data_train = torch.stack(data_train)
    torch.save(data_train,"./data/train_data.pth")