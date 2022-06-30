from pickletools import read_unicodestringnl
import torch
import torchvision
import numpy as np
import cv2
import os
import pandas as pd

#-------------------------------------------------------
# 1. Load design factor data
#-------------------------------------------------------
def load_design():
    design_raw = pd.read_csv(design_dir)
    design_raw.columns = ["no", 'CD', 'front overhang(%)', 'stagnation width(%)',
        'front corner roundness', 'side flat length(%)', 'side flat angle',
        'front vertical angle', 'height between stagnation to BLE(%)',
        'roof angle', 'half roof angle', 'end roof angle', 'rr glass angle',
        'rr angle', 'DLO boat tail angle', 'DLO rr corner roundness',
        'defusing angle']

    design = design_raw.copy()
    design.drop(['no'], axis=1, inplace=True)
    #designs = design.to_numpy()
    return design
#-------------------------------------------------------
# 2. Load flow field data
#-------------------------------------------------------
# 2-1. load flow field file name
def load_fieldcsvList():
    file_list = os.listdir(field_dir)
    csvlist = []
    for i in file_list:
        csv_name = os.path.splitext(i)[0]
        csvlist.append(csv_name)
    return csvlist
# 2-2. load flow field file
def load_fieldcsv(csv_name):
    field_raw = pd.read_csv(field_dir + csv_name + '.csv', header=None)
    field_raw.columns = ["Velocity_U", "Velocity_V","Velocity_W","Vorticity_Mag","X","Y","Z"]
    field = field_raw.copy()
    field.drop(["X","Y","Z"], axis=1, inplace=True)
    return field
# 2-3. load flow field files into memory as a ndarray
def load_fieldcsvs():
    csvlist = load_fieldcsvList()
    num_csv = len(csvlist)
    fields = np.zeros([num_csv, 4, 402, 602])
    for i, csvname in enumerate(csvlist):
        field = load_fieldcsv(csvname)
        vel_U = field[['Velocity_U']].to_numpy()
        vel_V = field[['Velocity_V']].to_numpy()
        vel_W = field[['Velocity_W']].to_numpy()
        vor_Mag = field[['Vorticity_Mag']].to_numpy()

        vel_U_to2d = np.flip(np.reshape(vel_U.T, (width, height)).T, axis=1)
        vel_V_to2d = np.flip(np.reshape(vel_V.T, (width, height)).T, axis=1)
        vel_W_to2d = np.flip(np.reshape(vel_W.T, (width, height)).T, axis=1)
        vor_Mag_to2d = np.flip(np.reshape(vor_Mag.T, (width, height)).T, axis=1)

        # cv2.imwrite(original_field_img_dir + 'U_{}.png'.format(i+1), vel_U_to2d)
        # cv2.imwrite(original_field_img_dir + 'V_{}.png'.format(i+1), vel_V_to2d)
        # cv2.imwrite(original_field_img_dir + 'W_{}.png'.format(i+1), vel_W_to2d)
        # cv2.imwrite(original_field_img_dir + 'Vor_{}.png'.format(i+1), vor_Mag_to2d)

        fields[i,0,:,:] = vel_U_to2d
        fields[i,1,:,:] = vel_V_to2d
        fields[i,2,:,:] = vel_W_to2d
        fields[i,3,:,:] = vor_Mag_to2d
    return fields
#-------------------------------------------------------
# 3. Get design difference data and save as csv
#-------------------------------------------------------
def mkdirs():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + 'Velocity_U', exist_ok=True)
    os.makedirs(output_dir + 'Velocity_V', exist_ok=True)
    os.makedirs(output_dir + 'Velocity_W', exist_ok=True)
    os.makedirs(output_dir + 'Vorticity_Mag', exist_ok=True)

def getDesDiff():
    designs = load_design()
    designDiff_list = []
    flow_idx = []
    for i in range(num_design-1):
        for j in range((i+1), num_design):
            # get design differences
            temp = designs.iloc[[i,j]].copy()
            temp_change = pd.DataFrame(temp.apply(lambda x:(len(pd.unique(x)) -1), axis=0))
            temp_change = temp_change.T
            designDiff_list.append(temp_change)
            # match each idxs to flow filed csv files
            if i >= 9:
                num1str = str(i+1)
            else:
                num1str = '0' + str(i+1)
            if j >= 9:
                num2str = str(j+1)
            else:
                num2str = '0' + str(j+1)
            flow_idx.append('flow_diff_{}_{}'.format(num1str, num2str))
    designDiff = pd.concat(designDiff_list)
    designDiff.insert(0, 'flow_idx', flow_idx, True)
    designDiff.to_csv(output_dir + 'design_diff.csv', index=False)
    return designDiff
#-------------------------------------------------------
# 4. Get flow differences data and save as images
#-------------------------------------------------------
def getFlowDiff():
    fields = load_fieldcsvs()
    num_diff = int(num_design * (num_design - 1) / 2)
    flowDiff = np.zeros([num_diff,4,402,602])
    k = 0
    for i in range(num_design-1):
        for j in range((i+1), num_design):
            # Make field difference images
            velUdiff = np.abs(fields[i,0,:,:] - fields[j,0,:,:])
            velVdiff = np.abs(fields[i,1,:,:] - fields[j,1,:,:])
            velWdiff = np.abs(fields[i,2,:,:] - fields[j,2,:,:])
            vorMagdiff = np.abs(fields[i,3,:,:] - fields[j,3,:,:])
            flowDiff[k,0,:,:] = velUdiff
            flowDiff[k,1,:,:] = velVdiff
            flowDiff[k,2,:,:] = velWdiff
            flowDiff[k,3,:,:] = vorMagdiff
            k += 1

            if i >= 9:
                num1str = str(i+1)
            else:
                num1str = '0' + str(i+1)
            if j >= 9:
                num2str = str(j+1)
            else:
                num2str = '0' + str(j+1)
            cv2.imwrite(output_dir + 'Velocity_U/' + 'diff_{}_{}.png'.format(num1str, num2str), velUdiff)
            cv2.imwrite(output_dir + 'Velocity_V/' + 'diff_{}_{}.png'.format(num1str, num2str), velVdiff)
            cv2.imwrite(output_dir + 'Velocity_W/' + 'diff_{}_{}.png'.format(num1str, num2str), velWdiff)
            cv2.imwrite(output_dir + 'Vorticity_Mag/' + 'diff_{}_{}.png'.format(num1str, num2str), vorMagdiff)

    print('Making flow difference images complete')
    return flowDiff

def mkDiff():
    flow_diff = getFlowDiff()
    design_diff = getDesDiff()
    return flow_diff, design_diff
#-------------------------------------------------------
# -. Methods for loading & saving diff_data
#-------------------------------------------------------
def loadDiff(diff_dir):
    designDiff_dir = diff_dir + 'design_diff.csv'
    flowDiff_dir = diff_dir
    designDiff = pd.read_csv(designDiff_dir)
    
    velUDiff_dir_list = os.listdir(flowDiff_dir + 'Velocity_U/')
    velVDiff_dir_list = os.listdir(flowDiff_dir + 'Velocity_V/')
    velWDiff_dir_list = os.listdir(flowDiff_dir + 'Velocity_W/')
    vorMagDiff_dir_list = os.listdir(flowDiff_dir + 'Vorticity_Mag/')
    num = len(velUDiff_dir_list)
    flowDiff = np.zeros([num,4,402,602])
    for n in range(num):
        velUdiff = cv2.imread(flowDiff_dir + 'Velocity_U/{}'.format(velUDiff_dir_list[n]), cv2.IMREAD_GRAYSCALE)
        velVdiff = cv2.imread(flowDiff_dir + 'Velocity_V/{}'.format(velVDiff_dir_list[n]), cv2.IMREAD_GRAYSCALE)
        velWdiff = cv2.imread(flowDiff_dir + 'Velocity_W/{}'.format(velWDiff_dir_list[n]), cv2.IMREAD_GRAYSCALE)
        vorMagdiff = cv2.imread(flowDiff_dir + 'Vorticity_Mag/{}'.format(vorMagDiff_dir_list[n]), cv2.IMREAD_GRAYSCALE)
        flowDiff[n,0,:,:] = velUdiff
        flowDiff[n,1,:,:] = velVdiff
        flowDiff[n,2,:,:] = velWdiff
        flowDiff[n,3,:,:] = vorMagdiff
    return flowDiff, designDiff
    
def getDiff(is_diff_exist=True):
    '''
    기존에 저장된 diff image file이 존재하는지에 따라서 다르게 데이터를 불러오는 메소드
    존재하지 않는다면 사용한 유동장 데이터와 디자인 데이터를 활용해 차이 데이터 이미지를 저장한다.
    * input *
    is_diff_exist : diff image file이 존재하지 않는 경우 False. 존재하면 True

    * output *
    is_diff_exist가 True인 경우,
        flow_diff : 이미지로 저장하기 전의 유동장 차이 데이터를 반환(ndarray : 2556 x 4 x 402 x 602)
        design_diff : 디자인 요소 차이 데이터 반환(dataframe)
    is_diff_exist가 False인 경우,
        flow_diff : 이미지로 저장한 후의 유동장 차이 데이터를 반환(ndarray : 2556 x 4 x 402 x 602)
        design_diff : 디자인 요소 차이 데이터 반환(dataframe)
    '''
    if is_diff_exist == True:
            flow_diff, design_diff = loadDiff(output_dir)
            return flow_diff, design_diff
    else:
        flow_diff, design_diff = mkDiff()
        return flow_diff, design_diff

def saveFlowDiff_NPY(flowDiff, name='flow_diff'):
    np.save(output_dir + name,flowDiff)
def loadFlowDiff_NPY(name='flow_diff'):
    flow_diff = np.load(output_dir + '{}.npy'.format(name))
    return flow_diff
#-------------------------------------------------------
# 5. Main
#-------------------------------------------------------
def loadData(b_dir, d_dir, f_dir, o_dir, h=402, w=602, num_d=72, num_typeD=15, if_exist=False):
    '''
    b_dir : 코드 실행 경로
    d_dir : 디자인 인자 데이터 저장 경로(case.csv)
    f_dir : 유동장 데이터 저장 경로(flow_XX.csv 파일 저장된 폴더)
    o_dir : 유동장 차이 데이터 저장 경로
    h : 유동장 데이터 이미지 높이=402
    w : 유동장 데이터 이미지 너비=602
    num_d : 디자인 인자 조합 수=72
    num_typeD : 디자인 인자 수=15
    if_exist : 유동장 차이 데이터가 이미 존재하는지 여부 체크
    '''
    global base_dir, design_dir, field_dir, height, width, num_design, design_typeNum, output_dir
    base_dir = b_dir
    design_dir = b_dir + d_dir
    field_dir = b_dir + f_dir
    output_dir = b_dir + o_dir
    height = h
    width = w
    num_design = num_d
    design_typeNum = num_typeD
    design_typeNum += 1

    os.chdir(base_dir)
    mkdirs()
    
    flow_diff, design_diff = getDiff(if_exist)
    return flow_diff, design_diff

class Flow():
    '''
    유동장 데이터를 관리하는 클래스
    자동으로 이미지 파일을 만들어 관리함

    ** input **
    flow_dir : 유동장 데이터 파일(csv)이 들어있는 디렉토리 경로
    design_dir : 디자인 인자 데이터 파일(csv) 경로
    img_dir : 유동장 이미지 파일이 들어있는 디렉토리 경로
    req_flow : 필요한 유동장 데이터(default=['u','v','w','vor'])
    '''
    def __init__(self, flow_dir, design_dir, img_dir):
        # 원본 파일(csv) 경로 데이터
        self.flow_dir = flow_dir
        self.design_dir = design_dir
        self.img_dir = img_dir
        # 디자인 인자 종류
        self.design_types
        self.mkimg()
        # 이미지 경로 없다면 생성
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.img_dir + 'Velocity_U/', exist_ok=True)
        os.makedirs(self.img_dir + 'Velocity_V/', exist_ok=True)
        os.makedirs(self.img_dir + 'Velocity_W/', exist_ok=True)
        os.makedirs(self.img_dir + 'Vorticity_Mag/', exist_ok=True)
    def __len__(self):
        return 
    def mkimg(self):
        return
    def get_design(self, index):
        return
    def get_u(self, index):
        return
    def get_v(self, index):
        return
    def get_w(self, index):
        return
    def get_vor(self, index):
        return
    def get_u_img(self, index):
        return
    def get_v_img(self, index):
        return
    def get_w_img(self, index):
        return
    def get_vor_img(self, index):
        return
    def get_designs(self):
        return self.design_types
    def get_u_img_dir(self, index):
        return
    def get_v_img_dir(self, index):
        return
    def get_w_img_dir(self, index):
        return
    def get_vor_img_dir(self, index):
        return

class FlowDiff(Flow):
    '''
    유동장 차이 데이터를 관리하는 클래스
    '''
    def __init__(self, flow_diff_dir='', design_diff_dir=''):
        super(FlowDiff, self).__init__()
        self.flow_diff_dir = flow_diff_dir
        self.design_diff_dir = design_diff_dir

        self.u_diff
        self.v_diff
        self.w_diff
        self.vor_diff

        self.u_diff_dir
        self.v_diff_dir
        self.w_diff_dir
        self.vor_diff_dir

    def __len__(self):
        return
