############### predit ###################
from test_util import *
if __name__ == '__main__':
    with torch.no_grad():        
        ## load trained network 
        net = IR2Unet()
        net = nn.DataParallel(net)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(torch.load('model_IR2Unet.pth'),False)
        net.to(device)
        net.eval()

        input_path = 'lfs1.nii'
        output_path = 'chi1.nii'
        process_nii_file(input_path, output_path, net, device)
        print('end')
