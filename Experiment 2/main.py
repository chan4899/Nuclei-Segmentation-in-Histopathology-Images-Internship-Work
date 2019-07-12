from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import torchvision.transforms.functional as F

import torchvision
from torchvision import transforms

from networks.cenet import CE_Net_
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from Visualizer import Visualizer
from PIL import Image

import Constants
import image_utils

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def CE_Net_Train():
    NAME = 'CE-Net' + Constants.ROOT.split('/')[-1]

    # run the Visdom
    viz = Visualizer(env=NAME)

    solver = MyFrame(CE_Net_, dice_bce_loss, 2e-4)
    print("count",Constants.BATCHSIZE_PER_CARD)
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    print("batchsize",batchsize)

    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    dataset = ImageFolder(root_path=Constants.ROOT, datasets='Cell')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)


    dataset_val = ImageFolder(root_path='./test_data/DRIVE_dot_dash_training', datasets='Cell')
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=True,
        num_workers=4)


    # start the logging files
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()

    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0

        for img, mask in data_loader_iter:
            # solver.load('./weights/' + NAME + '.th')
            # print("iterating the dataloader")
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index = index + 1

        # show the original images, predication and ground truth on the visdom.
        show_image = (img + 1.6) / 3.2 * 255.
        viz.img(name='images', img_=show_image[0, :, :, :])
        viz.img(name='labels', img_=mask[0, :, :, :])
        viz.img(name='prediction', img_=pred[0, :, :, :])
        

        torchvision.utils.save_image(img[0, :, :, :], "images/image_"+str(epoch) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        torchvision.utils.save_image(mask[0, :, :, :], "images/mask_"+str(epoch) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
        torchvision.utils.save_image(pred[0, :, :, :], "images/pred_"+str(epoch) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)



        # x = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.uint8)
        # x = show_image[0,:,:,:]
        # print(x.shape)
        # pil_im = transforms.ToPILImage(mode = 'RGB')(x)
        # pil_im.save('/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/image_' + str(epoch) +  '.jpg')

        # x = mask[0,:,:,:]
        # print(x.shape)
        # pil_im = transforms.ToPILImage(mode = 'L')(x)
        # pil_im.save('/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/mask_' + str(epoch) +  '.jpg')

        # x = pred[0,:,:,:]
        # print(x.shape)
        # pil_im = transforms.ToPILImage(mode = 'HSV')(x.detach().cpu().numpy())
        # pil_im.save('/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/prediction_' + str(epoch) +  '.jpg')
        # (x.detach().numpy()).save("/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/image_" + str(epoch) + ".png")
        # cv2.imwrite('imagename.jpg', x.detach().numpy().astype('uint8')).transpose(2,1,0)
        # x = mask[0,:,:,:]
        # # F.to_pil_image(x.detach().numpy()).save("/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/mask_" + str(epoch) + ".png")
        # x = pred[0,:,:,:]
        # print(x.shape)
        # cv2.imwrite('imagename2.jpg', x.detach().numpy().astype('uint8'))
        
        # F.to_pil_image(x.detach().numpy()).save("/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/prediction_" + str(epoch) + ".png")
        print("saving images")
        print("Train_loss_for_all ", train_epoch_loss)
        print("length of (data_loader_iter) ", len(data_loader_iter))
        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        print(mylog, '********')
        print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        print(mylog, 'train_loss:', train_epoch_loss)
        print(mylog, 'SHAPE:', Constants.Image_size)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', Constants.Image_size)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            print("Saving the Weights")
            solver.save('./weights/' + NAME + '.th')
            if epoch%100==0:
                solver.save('./weights/' + NAME + str(epoch) + '.th')
        if no_optim > Constants.NUM_EARLY_STOP:
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
        mylog.flush()


        if (epoch%1==0):
            # validation save image
            print('in VALIDATION')
            # for 
            data_loader_iter_val = iter(data_loader_val)
            train_epoch_loss = 0
            index = 0

            for img, mask in data_loader_iter_val:
                # solver.load('./weights/' + NAME + '.th')
                solver.set_input(img, mask)

                train_loss, pred = solver.optimize_test()
                train_epoch_loss += train_loss
                index = index + 1
                # torchvision.utils.save_image(img[0, :, :, :], "test_data/results2/image_"+str(epoch) + '_' + str(index) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
                # torchvision.utils.save_image(mask[0, :, :, :], "test_data/results2/mask_"+str(epoch) + '_' + str(index) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
                # torchvision.utils.save_image(pred[0, :, :, :], "test_data/results2/pred_"+str(epoch) + '_' + str(index) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
            print("Train_loss_for_all ", train_epoch_loss)
            print("length of (data_loader_iter_val) ", len(data_loader_iter_val))
            print(train_epoch_loss/len(data_loader_iter_val))
            print('++++++++++++++++++++++++++++++++++')
            # show the original images, predication and ground truth on the visdom.
            # show_image = (img + 1.6) / 3.2 * 255.
            # viz.img(name='images', img_=show_image[0, :, :, :])
            # viz.img(name='labels', img_=mask[0, :, :, :])
            # viz.img(name='prediction', img_=pred[0, :, :, :])
            

            torchvision.utils.save_image(img[0, :, :, :], "test_data/results4/image_"+str(epoch) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
            torchvision.utils.save_image(mask[0, :, :, :], "test_data/results4/mask_"+str(epoch) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)
            torchvision.utils.save_image(pred[0, :, :, :], "test_data/results4/pred_"+str(epoch) + ".jpg", nrow=1, padding=2, normalize=True, range=None, scale_each=False, pad_value=0)



            # x = torch.tensor([[1,2,3],[4,5,6]], dtype = torch.uint8)
            # x = show_image[0,:,:,:]
            # print(x.shape)
            # pil_im = transforms.ToPILImage(mode = 'RGB')(x)
            # pil_im.save('/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/image_' + str(epoch) +  '.jpg')

            # x = mask[0,:,:,:]
            # print(x.shape)
            # pil_im = transforms.ToPILImage(mode = 'L')(x)
            # pil_im.save('/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/mask_' + str(epoch) +  '.jpg')

            # x = pred[0,:,:,:]
            # print(x.shape)
            # pil_im = transforms.ToPILImage(mode = 'HSV')(x.detach().cpu().numpy())
            # pil_im.save('/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/prediction_' + str(epoch) +  '.jpg')
            # (x.detach().numpy()).save("/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/image_" + str(epoch) + ".png")
            # cv2.imwrite('imagename.jpg', x.detach().numpy().astype('uint8')).transpose(2,1,0)
            # x = mask[0,:,:,:]
            # # F.to_pil_image(x.detach().numpy()).save("/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/mask_" + str(epoch) + ".png")
            # x = pred[0,:,:,:]
            # print(x.shape)
            # cv2.imwrite('imagename2.jpg', x.detach().numpy().astype('uint8'))
            
            # F.to_pil_image(x.detach().numpy()).save("/home/videsh/Downloads/Chandan/paper_implementation/CE-Net-master/images/prediction_" + str(epoch) + ".png")
            # print("saving images")

            # train_epoch_loss = train_epoch_loss/len(data_loader_iter)
            # print(mylog, '********')
            # print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
            # print(mylog, 'train_loss:', train_epoch_loss)
            # print(mylog, 'SHAPE:', Constants.Image_size)
            # print('********')
            # print('epoch:', epoch, '    time:', int(time() - tic))
            # print('train_loss:', train_epoch_loss)
            # print('SHAPE:', Constants.Image_size)

            # if train_epoch_loss >= train_epoch_best_loss:
            #     no_optim += 1
            # else:
            #     no_optim = 0
            #     train_epoch_best_loss = train_epoch_loss
            #     solver.save('./weights/' + NAME + '.th')
            # if no_optim > Constants.NUM_EARLY_STOP:
            #     print(mylog, 'early stop at %d epoch' % epoch)
            #     print('early stop at %d epoch' % epoch)
            #     break
            # if no_optim > Constants.NUM_UPDATE_LR:
            #     if solver.old_lr < 5e-7:
            #         break
            #     solver.load('./weights/' + NAME + '.th')
            #     solver.update_lr(2.0, factor=True, mylog=mylog)
            # mylog.flush()








    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    CE_Net_Train()



