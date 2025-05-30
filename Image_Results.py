import os
import cv2 as cv
import numpy as np


def Image_Results():
    Morphed = np.load('morph.npy', allow_pickle=True)
    Removed = np.load('removed.npy', allow_pickle=True)
    Occ = np.load('occ.npy', allow_pickle=True)
    Occ_removed = np.load('occ_removed.npy', allow_pickle=True)

    for i in range(6):
        if i == 5:
            Morphed = np.load('morph_Maruthi.npy', allow_pickle=True)
            Removed = np.load('removed_Maruthi.npy', allow_pickle=True)
            cv.imwrite('./Image_Results/Image_Results_Morphing/Morphed-' + str(i + 1) + '.jpg', Morphed)
            cv.imwrite('./Image_Results/Image_Results_Morphing/Removed-' + str(i + 1) + '.jpg', Removed)
            rem = cv.resize(Removed, [531, 413])
            # im_h = cv.hconcat([Morphed[i], Removed[i]])
            # # show the output image
            cv.imshow('Morphed', Morphed)
            cv.waitKey(0)

            cv.imshow('Removed', rem)
            cv.waitKey(0)
        else:

            cv.imwrite('./Image_Results/Image_Results_Morphing/Morphed-' + str(i + 1) + '.jpg', Morphed[i])
            cv.imwrite('./Image_Results/Image_Results_Morphing/Removed-' + str(i + 1) + '.jpg', Removed[i])
            rem = cv.resize(Removed[i], [531, 413])
            # im_h = cv.hconcat([Morphed[i], Removed[i]])
            # # show the output image
            cv.imshow('Morphed', Morphed[i])
            cv.waitKey(0)

            cv.imshow('Removed', rem)
            cv.waitKey(0)

    for i in range(6):
        if i == 5:
            Occ = np.load('occ_Maruthi.npy', allow_pickle=True)
            Occ_removed = np.load('occ_removed_Maruthi.npy', allow_pickle=True)
            cv.imwrite('./Image_Results/Image_Results_Occlusion/Occ-' + str(i + 1) + '.jpg', Occ)
            cv.imwrite('./Image_Results/Image_Results_Occlusion/Removed-' + str(i + 1) + '.jpg', Occ_removed)

            cv.imshow('Morphed', Occ)
            cv.waitKey(0)

            cv.imshow('Removed', Occ_removed)
            cv.waitKey(0)
        else:

            cv.imwrite('./Image_Results/Image_Results_Occlusion/Occ-' + str(i + 1) + '.jpg', Occ[i])
            cv.imwrite('./Image_Results/Image_Results_Occlusion/Removed-' + str(i + 1) + '.jpg', Occ_removed[i])

            cv.imshow('Morphed', Occ[i])
            cv.waitKey(0)

            cv.imshow('Removed', Occ_removed[i])
            cv.waitKey(0)

if __name__ == '__main__':
    Image_Results()
