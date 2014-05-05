import cv2
import numpy as np
from fn import _

def cut(img, bs):
    res = []
    for i in xrange(0, img.shape[0], bs):
        for j in xrange(0, img.shape[1], bs):
            res.append(img[i:i+bs, j:j+bs])
    return filter(_.shape == (bs, bs), res)

if __name__ == '__main__':
    img = cv2.imread('/home/daiver/BSR/BSDS500/data/groundImages/test/100039.mat_2.png',0)
    patches = cut(img, 16)
    neg = filter(lambda x:np.std(x) == 0.0, patches)
    pos = filter(lambda x:np.std(x) >  0.0, patches)
    print len(neg), len(pos)
    for i, x in enumerate(neg[:20]):
        cv2.imwrite('/home/daiver/dump2/neg-%d.png' % i, x)
    j = 0
    for i, x in enumerate(pos):
        cv2.imshow('', cv2.pyrUp(cv2.pyrUp(cv2.pyrUp(x)))*10)
        print np.std(x)
        k = cv2.waitKey() % 0x100
        print k
        if k == 32:
            j += 1
            cv2.imwrite('/home/daiver/dump2/pos-%d.png' % j, x)
        if j > 19: break
