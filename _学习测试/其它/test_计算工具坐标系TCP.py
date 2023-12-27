import numpy as np
import pandas as pd
import pysnooper
import math
from scipy.spatial.transform import Rotation as R

class tool_cal():
    def __init__(self):
        """
		load data from csv
		tool_points(0~5) : use robot effectors to touch the same points in the world
                      and record the pos 
        tool_poses_tran(0~3):count tanslation
        tool_poses_rot(3~5):count rotation
		"""
        #with open("tool_data.csv") as file:
        #    tool_poses = pd.read_csv(file, header = None)
        #    tool_poses = np.array(tool_poses)
        tool_poses = np.array([
            [   873.345,   -412.269,    -442.898,    -15.524,    25.052,     -43.023],
            [   902.044,   -55.660,     -531.593,    -6.280,     20.533,     17.568],
            [   872.361,   86.848,      -429.534,    67.371,     25.366,     46.094],
            [   826.916,   -127.867,    -508.869,    6.161,      32.620,     6.981],
            [   1303.980,  -127.867,    -508.869,    6.161,  32.620,   6.981],
            [   826.916,   -127.867,    -198.024,    6.161,  32.620,   6.981]
        ])

        
        # cal translation
        self.tran_tran=[]
        self.tran_rotm=[]
        tool_poses_tran = tool_poses[0:4,:]
        for pose in tool_poses_tran:
            # set translation
            self.tran_tran.append(np.array([[pose[0]],[pose[1]],[pose[2]]]))
            
            # set rotation
            r = R.from_euler('xyz', np.array([pose[3], pose[4], pose[5]]), degrees=True)
            self.tran_rotm.append(r.as_matrix())

        tool_tran = self.cal_tran()
        
        # cal rotation
        self.rot_tran=[]
        self.rot_rotm=[]
        tool_poses_rot = tool_poses[3:6,:]
        for pose in tool_poses_rot:
            # set translation
            self.rot_tran.append(np.array([[pose[0]],[pose[1]],[pose[2]]]))
            
            # set rotation
            r = R.from_euler('xyz', np.array([pose[3], pose[4], pose[5]]), degrees=True)
            self.rot_rotm.append(r.as_matrix())

        tool_rot=self.cal_rotm(tool_tran)

        # get transformation
        tool_T = np.array(np.zeros((4,4)))
        tool_T[0:3,0:3] = tool_rot
        tool_T[0:3,3:] = tool_tran
        tool_T[3:,:] = [0,0,0,1]

        print("tool_T = ")
        print(tool_T)

        # change into quat
        q = R.from_matrix(tool_rot)
        print("quat = ")
        print(q.as_quat())



    def cal_tran(self):
        # 公式五，拆分后计算
        # tran_data=[]
        # rotm_data=[]
        # for i in range(len(self.tran_tran)-1):
        #     tran_data.append(self.tran_tran[i+1] - self.tran_tran[i])
        #     rotm_data.append(self.tran_rotm[i] - self.tran_rotm[i+1])
        
        # L = np.array(np.zeros((3,3)))
        # R = np.array(np.zeros((3,1)))
        # for i in range(len(tran_data)):
        #     L = L + np.dot(rotm_data[i],rotm_data[i])
        #     R = R + np.dot(rotm_data[i],tran_data[i])
        # print(np.linalg.inv(L).dot(R))
        # tool_tran = np.linalg.inv(L).dot(R)
		
        # 构建Ax=B
        A = self.tran_rotm[0] - self.tran_rotm[1]
        B = self.tran_tran[1] - self.tran_tran[0]
        for i in range(1,len(self.tran_tran)-1):
            A = np.vstack((A,self.tran_rotm[i] - self.tran_rotm[i+1]))
            B = np.vstack((B,self.tran_tran[i+1] - self.tran_tran[i]))
        
        # 广义逆
        tool_tran = np.linalg.pinv(A).dot(B)
        
        # svd
        u,s,v = np.linalg.svd(A,0)
        c = np.dot(u.T,B)
        w = np.linalg.solve(np.diag(s),c)
        tool_tran = np.dot(v.T,w)

        return tool_tran

    def cal_rotm(self, tran):
        # centre
        P_otcp_To_B = np.dot(self.rot_rotm[0], tran) + self.rot_tran[0]

        # cal the dircction vector of x
        P_xtcp_To_B = np.dot(self.rot_rotm[1],tran) + self.rot_tran[1]
        vector_X = P_xtcp_To_B - P_otcp_To_B
        dire_vec_x_o = np.linalg.inv(self.rot_rotm[0]).dot(vector_X) / np.linalg.norm(vector_X)

        # cal the dircction vector of z
        P_ztcp_To_B = np.dot(self.rot_rotm[2],tran) + self.rot_tran[2]
        vector_Z = P_ztcp_To_B - P_otcp_To_B
        dire_vec_z_o = np.linalg.inv(self.rot_rotm[0]).dot(vector_Z) / np.linalg.norm(vector_Z)

        # cal the dircction vector of y
        dire_vec_y_o = np.cross(dire_vec_z_o.T, dire_vec_x_o.T)

        # modify the dircction vector of z 
        dire_vec_z_o = np.cross(dire_vec_x_o.T, dire_vec_y_o)

        # cal rotation matrix
        tool_rot = np.array(np.zeros((3,3)))
        tool_rot[:,0] = dire_vec_x_o.T
        tool_rot[:,1] = dire_vec_y_o
        tool_rot[:,2] = dire_vec_z_o
        return tool_rot

if __name__ == "__main__":
    tool_cal()

    #{[[-362.3273544213903], [-75.38729713141531], [19.97264533356178]]}

    