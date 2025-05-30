import os
import numpy as np
import cv2 as cv
from CO import CO
from CSA import CSA
from GWO import GWO
from MAO import MAO
from ModeL_VGG16 import Model_VGG16
from Model_CNN import Model_CNN
from Model_DCNN import Model_DCNN
from Model_DTCN import Model_DTCN
from Model_TVGG16_AM import Model_TVGG16_AM
from Morphing import Match_detection
from Global_vars import Global_vars
from numpy import matlib
from random import uniform
from Objective_Function_1 import Objective_Function_1
from Objective_Function_2 import Objective_Function_2
from PROPOSED import PROPOSED
from AGANI import AGANI
from Model_TVGG16_AM_Feat import Model_TVGG16_AM_Feat
from Image_Results import Image_Results
from Plot_Results import Plot_Convergence, Plot_Results, Plot_Confusion, Plot_ROC

## READ DATASETS
an = 0
if an == 1:
    Target_Detection = []
    Target_Recognition = []
    Images = []
    Path = './CelebA_Spoof/Data/train/'
    dir = os.listdir(Path)
    for i in range(len(dir)):
        print(i, len(dir))
        sub_dir = os.listdir(Path + dir[i])
        for j in range(len(sub_dir)):
            if 'OneNote' not in sub_dir[j]:
                sub_dir__ = os.listdir(Path + dir[i] + '/' + sub_dir[j])
                for k in range(len(sub_dir__)):
                    if '.txt' not in sub_dir__[k]:
                        if 'OneNote' not in sub_dir__[k]:
                            image = cv.imread(Path + dir[i] + '/' + sub_dir[j] + '/' + sub_dir__[k])
                            image = cv.resize(image, [128, 128]) / 255.0
                            Images.append(image)
                            Target_Detection.append(j)
                            Target_Recognition.append(i + 1)
    uni = np.unique(np.asarray(Target_Recognition))
    Target = np.zeros((len(Target_Recognition), len(uni)), dtype=int)
    for i in range(len(uni)):
        ind = np.where(Target_Recognition == uni[i])
        Target[ind[0], i] = 1
    # Read Morphed and Occluded images
    Morphed_Image = []
    Morphed_Target = []
    Occluded = []
    Path = './Dataset/'
    dir = os.listdir(Path)
    for i in range(len(dir)):
        print(i, len(dir))
        sub_dir = os.listdir(Path + dir[i])
        for j in range(len(sub_dir)):
            sub_dirs = os.listdir(Path + dir[i] + '/' + sub_dir[j])
            for k in range(len(sub_dirs)):
                image = cv.imread(Path + dir[i] + '/' + sub_dir[j] + '/' + sub_dirs[k])
                image = cv.resize(image, [128, 128]) / 255.0
                if i == 0:
                    Morphed_Image.append(image)
                    Morphed_Target.append(j)
                else:
                    Occluded.append(image)
    Morphed_Target = [x ^ 1 for x in Morphed_Target]
    np.save('Morphed_Image.npy', Morphed_Image)
    np.save('Morphed_Target.npy', Morphed_Target)
    np.save('Occluded.npy', Occluded)
    np.save('Images.npy', Images)
    np.save('Target_Detection.npy', Target_Detection)
    np.save('Target_Recognition.npy', Target)
#
## Append all Face images
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Target = np.load('Target_Detection.npy', allow_pickle=True)
    Morphed = np.load('Morphed_Image.npy', allow_pickle=True)
    Occluded = np.load('Occluded.npy', allow_pickle=True)
    Morph_Target = np.load('Morphed_Target.npy', allow_pickle=True)
    Tot_Images = np.concatenate((Images, Morphed, Occluded), axis=0)
    Tot_Target = np.concatenate((Target, Morph_Target, np.ones((len(Occluded))).astype('int')), axis=0)
    np.save('Tot_Images.npy', Tot_Images)
    np.save('Tot_Target.npy', Tot_Target)

## FACE ATTACK DETECTION using TVGG16-AM
an = 1
if an == 1:
    Images = np.load('Tot_Images.npy', allow_pickle=True)
    Target = np.load('Tot_Target.npy', allow_pickle=True).reshape(-1, 1)

    learnper = round(Images.shape[0] * 0.75)
    train_data = Images[:learnper, :]
    train_target = Target[:learnper, :]
    test_data = Images
    test_target = Target

    predictions, Eval = Model_TVGG16_AM(train_data, train_target, test_data, test_target)
    ind1 = np.where(predictions == 1)
    Fake_Images = Images[ind1[0]]
    ind2 = np.where(predictions == 0)
    Real_Images = Images[ind2[0]]
    np.save('Fake_Images.npy', Fake_Images)
    np.save('Real_Images.npy', Real_Images)
    np.save('Ind1.npy', ind1[0])
    np.save('Ind2.npy', ind2[0])

## Remove Morphing by master face detection
an = 0
if an == 1:
    Morphed = []
    Real_Occ = []
    Fake_Images = np.load('Fake_Images.npy', allow_pickle=True)
    Images = np.load('Images.npy', allow_pickle=True)
    Fake_Ind = np.load('Ind1.npy', allow_pickle=True)
    Target = np.load('Target_Recognition.npy', allow_pickle=True)
    cls = 0
    ind = np.where(Target[:, cls] == 1)
    Image_2 = Images[ind[0][0]]
    for i in range(len(Fake_Images)):
        print(i + 1, len(Fake_Images))
        Image_1 = Fake_Images[i]
        if Fake_Ind[i] != Fake_Ind[i - 1] + 1 and i != 0:
            cls += 1
            ind = np.where(Target[:, cls] == 1)
            Image_2 = Images[ind[0][0]]
        morphed = Match_detection(Image_1, Image_2)
        Morphed.append(morphed)
        Real_Occ.append(Image_2)
    np.save('Morphed_.npy', Morphed)
    np.save('Real_Occ.npy', Real_Occ)

## Optimization for Occlusion by face inpainting
an = 0
if an == 1:
    BestSol = []
    Fitness = []
    Fake_Images = np.load('Fake_Images.npy', allow_pickle=True)
    Real_Images = np.load('Real_Occ.npy', allow_pickle=True)
    Global_vars.Real_Images = Real_Images
    Global_vars.Fake_Images = Fake_Images

    Npop = 10
    Ch_len = 3
    xmin = matlib.repmat(np.concatenate([5, 5, 300], axis=None), Npop,
                         1)
    xmax = matlib.repmat(np.concatenate([255, 50, 1000], axis=None),
                         Npop,
                         1)
    initsol = np.zeros((xmax.shape))
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
    fname = Objective_Function_1
    Max_iter = 50

    print("MAO...")
    [bestfit1, fitness1, bestsol1, time1] = MAO(initsol, fname, xmin, xmax, Max_iter)

    print("CO...")
    [bestfit2, fitness2, bestsol2, time2] = CO(initsol, fname, xmin, xmax, Max_iter)

    print("GWO...")
    [bestfit3, fitness3, bestsol3, time3] = GWO(initsol, fname, xmin, xmax, Max_iter)

    print("CSA...")
    [bestfit4, fitness4, bestsol4, time4] = CSA(initsol, fname, xmin, xmax, Max_iter)

    print("Improved CSA...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    BestSol.append([bestsol1.ravel(), bestsol2.ravel(), bestsol3.ravel(), bestsol4.ravel(), bestsol5.ravel()])
    Fitness.append([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])
    np.save('Best_Sol_1.npy', BestSol)
    np.save('Fitness_1.npy', Fitness)

## Optimized Occlusion by face inpainting
an = 0
if an == 1:
    Fake_Images = np.load('Fake_Images.npy', allow_pickle=True)
    Real_Images = np.load('Real_Occ.npy', allow_pickle=True)
    BestSol = np.load('Best_Sol_1.npy', allow_pickle=True)

    Images = AGANI(Real_Images, Fake_Images, BestSol[0][4])
    np.save('Face_Inpainted.npy', Images)

## Combine the Images for recognition
an = 0
if an == 1:
    Target = np.load('Target_Detection.npy', allow_pickle=True)
    Tar_Recog = np.load('Target_Recognition.npy', allow_pickle=True)
    Real_Images = np.load('Real_Images.npy', allow_pickle=True)
    Face_Inpainted = np.load('Face_Inpainted.npy', allow_pickle=True)
    Morphed = np.load('Morphed.npy', allow_pickle=True)
    ind1 = np.where(Target == 0)
    ind2 = np.where(Target == 1)
    ind3 = np.where(Target == 1)
    Im = np.concatenate((Real_Images, Face_Inpainted, Morphed), axis=0)
    Target = np.concatenate((Tar_Recog[ind1], Tar_Recog[ind2], Tar_Recog[ind3]), axis=0)
    np.save('Im.npy', Im)
    np.save('Target.npy', Target)

## Features extracted from TVGG16-AM
an = 0
if an == 1:
    Images = np.load('Im.npy', allow_pickle=True)
    Deep_Features = Model_TVGG16_AM_Feat(Images)
    np.save('Deep_Features.npy', Deep_Features)

## Optimization for Face Recognition
an = 0
if an == 1:
    BestSol = []
    Fitness = []
    Features = np.load('Deep_Features.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_vars.Features = Features
    Global_vars.Target = Target

    Npop = 10
    Ch_len = 3
    xmin = matlib.repmat(np.concatenate([5, 5, 10], axis=None), Npop,
                         1)
    xmax = matlib.repmat(np.concatenate([255, 50, 100], axis=None),
                         Npop,
                         1)
    initsol = np.zeros((xmax.shape))
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
    fname = Objective_Function_2
    Max_iter = 50

    print("MAO...")
    [bestfit1, fitness1, bestsol1, time1] = MAO(initsol, fname, xmin, xmax, Max_iter)

    print("CO...")
    [bestfit2, fitness2, bestsol2, time2] = CO(initsol, fname, xmin, xmax, Max_iter)

    print("GWO...")
    [bestfit3, fitness3, bestsol3, time3] = GWO(initsol, fname, xmin, xmax, Max_iter)

    print("CSA...")
    [bestfit4, fitness4, bestsol4, time4] = CSA(initsol, fname, xmin, xmax, Max_iter)

    print("Improved CSA...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    BestSol.append([bestsol1.ravel(), bestsol2.ravel(), bestsol3.ravel(), bestsol4.ravel(), bestsol5.ravel()])
    Fitness.append([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])
    np.save('Best_Sol_2.npy', BestSol)
    np.save('Fitness_2.npy', Fitness)

## Face Recognition - Cross Validation
an = 0
if an == 1:
    Eval_all = []
    Features = np.load('Deep_Features.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    soln = np.load('Best_Sol_2.npy', allow_pickle=True)
    k_fold = 5  # K fold
    for m in range(k_fold):
        EVAL = np.zeros((10, 14))
        for i in range(5):  # for all algorithms
            sol = soln[i].astype('int')
            Total_Index = np.arange(Features.shape[0])
            Test_index = np.arange(((m + 1 - 1) * (Features.shape[0] / k_fold)) + 1,
                                   (m + 1) * (Features.shape[0] / k_fold))
            Train_Index = np.setdiff1d(Total_Index, Test_index)
            train_data = Features[Train_Index, :]
            train_target = Target[Train_Index, :]
            test_data = Features[Test_index, :]
            test_target = Target[Test_index, :]
            EVAL[i, :] = Model_DTCN(train_data, train_target, test_data, test_target,
                                    sol)  # Adaptive Deep Temporal Convolution Network
        EVAL[5, :] = Model_CNN(train_data, train_target, test_data, test_target)  # Model CNN
        EVAL[6, :] = Model_VGG16(train_data, train_target, test_data, test_target)  # Model VGG16
        EVAL[7, :] = Model_DCNN(train_data, train_target, test_data, test_target)  # Model DCNN
        EVAL[8, :] = Model_DTCN(train_data, train_target, test_data, test_target,
                                np.asarray([100, 25, 50]))  # Deep Temporal Convolution Network
        EVAL[9, :] = EVAL[4, :]
        Eval_all.append(EVAL)
    np.save('Eval_all.npy', Eval_all)

Image_Results()
Plot_Convergence()
Plot_Results()
Plot_Confusion()
Plot_ROC()
