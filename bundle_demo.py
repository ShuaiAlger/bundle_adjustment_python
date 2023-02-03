
import numpy as np
import math
import cv2

np.set_printoptions(precision = 6, suppress=True)





def p2d_to_homo(p2d):
    p2dh = np.concatenate([p2d, np.ones(p2d.shape[0], 1)], axis=1)
    return p2dh

def rodrigues2rot(rodrigues):
    rot = cv2.Rodrigues(rodrigues)[0]
    return rot

# def rot2rodrigues(rot):
#     rodrigues = cv2.Rodrigues(rot)
#     return rodrigues

# class PinholeCamera:
#     def __init__(self, params=[520.9, 521.0, 325.1, 249.7, 0, 0, 0, 0, 0]):
        # self.fx = params[0]
        # self.fy = params[1]
        # self.cx = params[2]
        # self.cy = params[3]
        # self.k1 = params[4]
        # self.k2 = params[5]
        # self.p1 = params[6]
        # self.p2 = params[7]
        # self.k3 = params[8]
        # self.intri = np.asarray([[ self.fx,       0, self.cx ],
        #                          [       0, self.fy, self.cy ],
        #                          [       0,       0,       1 ]])
    # def proj3d_to_2d(self, p3ds):
    #     p2dsh = self.intri @ p3ds
    #     return p2dsh

fx = 520.9
fy = 521.0
cx = 325.1
cy = 249.7


class Optimizer:
    def __init__(self, method="lm"):
        self.method = method
        self.USE_LDLT = True
        self.USE_LM = False

        if method == "lm":
            self.USE_LM = True
        self.lam = 0.001

        self.iterations = 10

        self.points3d = None
        self.points2d = None
        self.vertex_poses = None

        # self.fx = 520.9
        # self.fy = 521.0
        # self.cx = 325.1
        # self.cy = 249.7

    def read_data(self, p3ds, p2ds, poses):
        self.points3d = p3ds
        self.points2d = p2ds
        self.vertex_poses = poses

    def cal_J(self):
        pose_num = self.vertex_poses.shape[0]
        point_num = self.points3d.shape[0]
        J_all = np.zeros((2*pose_num*point_num, 6*pose_num))
        J = np.zeros((2, 6))
        for j in range(pose_num):
            for i in range(point_num):
                R = self.vertex_poses[j, 0:3, 0:3]
                t = self.vertex_poses[j, 0:3, 3]
                # P = R @ self.points3d[i] + t
                P = R @ self.points3d[i].T + t.T

                x = P[0]
                y = P[1]
                z = P[2]

                z2 = z * z
                J[0, 0] = -1.0 / z * fx
                J[0, 1] = 0
                J[0, 2] = x / z2 * fx
                J[0, 3] = x * y / z2 * fx
                J[0, 4] = -(1 + (x * x / z2)) * fx
                J[0, 5] = y / z * fx
                J[1, 0] = 0
                J[1, 1] = -1.0 / z * fy
                J[1, 2] = y / z2 * fy
                J[1, 3] = (1 + (y * y / z2)) * fy
                J[1, 4] = -x * y / z2 * fy
                J[1, 5] = -x / z * fy

                row = 2 * j * point_num + 2 * i
                col = 6 * j

                J_all[row:row+2, col:col+6] = J
        return J_all

    def cal_reproj_err(self):
        pose_num = self.vertex_poses.shape[0]
        point_num = self.points3d.shape[0]
        reproj_err = 0.0
        ReprojErr = np.zeros((2 * pose_num * point_num, 1))
        for j in range(pose_num):
            for i in range(point_num):
                R = self.vertex_poses[j, 0:3, 0:3]
                t = self.vertex_poses[j, 0:3, 3]
                # P = R @ self.points3d[i] + t
                P = R @ self.points3d[i].T + t.T

                x = P[0]
                y = P[1]
                z = P[2]
                p_u = fx * x / z + cx
                p_v = fy * y / z + cy
                du = self.points2d[i, 0] - p_u
                dv = self.points2d[i, 1] - p_v

                ReprojErr[j * point_num * 2 + 2 * i] = du
                ReprojErr[j * point_num * 2 + 2 * i + 1] = dv
                reproj_err = reproj_err + (du*du + dv*dv)
        if self.USE_LM:
            return reproj_err / (pose_num * point_num), ReprojErr
        else:
            return reproj_err / (pose_num * point_num), ReprojErr

    
    def cal_se3(self, J, ReprojErr):
        H = J.T @ J
        if self.USE_LM:
            H = H + self.lam * np.eye(H.shape[0])
        if self.USE_LDLT:
            g = -J.T @ ReprojErr
            delta_x = np.linalg.solve(H, g)
        else:
            minus_b = -1.0 * ReprojErr
            delta_x = np.linalg.inv(H) @ J.T @ minus_b
        return delta_x

    def update_se3(self, delta_se3):
        pose_num = self.vertex_poses.shape[0]
        for j in range(pose_num):
            r = delta_se3[3:6]
            t = delta_se3[0:3]
            rot = rodrigues2rot(r)
            dT = np.zeros((4, 4))
            dT[0:3, 0:3] = rot
            dT[0:3, 3:4] = t
            dT[3, 3] = 1
            self.vertex_poses[j] = dT @ self.vertex_poses[j]

    def gauss_newton(self):
        for i in range(self.iterations):
            print(self.vertex_poses)
            J = self.cal_J()
            cur_err, ReprojErr = self.cal_reproj_err()
            print("cur_err : ", cur_err)
            delta_se3 = self.cal_se3(J, ReprojErr)
            # print(delta_se3)
            if np.linalg.norm(delta_se3) < 0.00001:
                break
            self.update_se3(delta_se3)

    def levenburg(self):
        for i in range(self.iterations):
            print(self.vertex_poses)
            J = self.cal_J()

            # rho = dReprojErr/J

            cur_err, ReprojErr = self.cal_reproj_err()
            print("cur_err : ", cur_err)
            delta_se3 = self.cal_se3(J, ReprojErr)
            # print(delta_se3)
            if np.linalg.norm(delta_se3) < 0.00001:
                break
            self.update_se3(delta_se3)


def read_data():
    p2dtxt = "p2d.txt"
    p3dtxt = "p3d.txt"
    with open(p2dtxt) as f:
        lines = f.readlines()
        p2ds = np.zeros((len(lines), 2))
        for i in range(len(lines)):
            u = lines[i].split('\n')[0].split(' ')[0]
            v = lines[i].split('\n')[0].split(' ')[1]
            p2ds[i, 0] = u
            p2ds[i, 1] = v
    with open(p3dtxt) as f:
        lines = f.readlines()
        p3ds = np.zeros((len(lines), 3))
        for i in range(len(lines)):
            x = lines[i].split('\n')[0].split(' ')[0]
            y = lines[i].split('\n')[0].split(' ')[1]
            z = lines[i].split('\n')[0].split(' ')[2]
            p3ds[i, 0] = x
            p3ds[i, 1] = y
            p3ds[i, 2] = z
    return p2ds, p3ds



if __name__ == '__main__':
    p2ds, p3ds = read_data()

    pose = np.eye(4, 4)
    poses = np.zeros((1, 4, 4))
    poses[0] = pose

    opt = Optimizer()
    opt.read_data(p3ds, p2ds, poses)
    opt.gauss_newton()

