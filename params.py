import numpy as np
import warp


class Params:
    def __init__(self, args):
        self.warpScale = {"pert": 0.25, "trans": 0.25}
        self.warpType = args.warpType
        self.batchSize = args.batchSize
        self.baseLR, self.baseLRST = args.lr, args.lrST
        # --- below are automatically set ---
        self.H, self.W = 28, 28
        self.visBlockSize = 12
        self.GPUdevice = "/gpu:{0}".format(args.gpu)
        self.pDim = 2 if self.warpType == "translation" else \
            4 if self.warpType == "similarity" else \
                6 if self.warpType == "affine" else \
                    8 if self.warpType == "homography" else None
        self.canon4pts = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float32)
        self.Im4pts = np.array([[0, 0], [0, self.H - 1], [self.W - 1, self.H - 1], [self.W - 1, 0]], dtype=np.float32)
        self.warpGTmtrx = warp.fit(self.canon4pts, self.Im4pts, "affine")
        if args.type == "CNN": self.baseLRST = 0
