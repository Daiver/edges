import os, cv2, numpy

main_dir_path = '/home/daiver/BSR/BSDS500/data/groundImages/'
target_dir_path = '/home/daiver/BSR/BSDS500/data/groundEdges/'
for dir_name in os.listdir(main_dir_path):
    for fname in os.listdir(os.path.join(main_dir_path, dir_name)):
        img = cv2.imread(os.path.join(main_dir_path, dir_name, fname), 0)
        print dir_name, fname
        img2 = cv2.Canny(img, 1, 1, 3)
        #img4 = cv2.Canny(img, 0, 1, ksize=51)
        cv2.imshow('', img2)
        cv2.imshow('orig', img)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(target_dir_path, dir_name, fname), img2)
