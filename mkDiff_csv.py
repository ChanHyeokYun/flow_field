<<<<<<< HEAD
''' 
mkDiff_csv.py
save u,v,w,vorticity differences and design differences

Maker info.
Name                : Yun ChanHyeok
Last update date    : 05/23/2022
'''
import numpy as np
import pandas as pd
import os
import cv2
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

if __name__ == '__main__':
    global base_dir, design_dir, field_dir, height, width, num_design, design_typeNum, output_dir
    #global original_field_img_dir

    # 데이터 위치 지정
    base_dir = 'C:/Users/user/Desktop/code/hyundai/'
    design_dir = base_dir + 'case.csv'
    field_dir = base_dir + 'flowRearranged/'
    output_dir = base_dir + 'dataset/'
    height = 402
    width = 602
    num_design = 72
    design_typeNum = 16
    # 유동장 데이터 검토용
    # original_field_img_dir = output_dir + 'ori_field_img/'
    # os.makedirs(original_field_img_dir, exist_ok=True)

    os.chdir(base_dir)
    mkdirs()

    '''
    Case 1. diff data file이 없을 때
        1-1. 유동장 차이 데이터는 image로 저장하고, 디자인 차이 데이터는 csv로 저장
            * output * (getDiff)
            flow_diff : 2556 x 4 x 402 x 602 ndarray
            design_diff : Dataframe type
                1열 : idx1 : flow_diff_{}_{}의 형태로, {}번째 디자인과 {}번쨰 디자인을 비교한 경우인지를 표시.
                2열 : CD
                3열~17열 : 디자인요소
    '''
    flow_diff, design_diff = getDiff(False)
    '''
        1-2. 이미지 파일이 아닌 .npy파일로 ndarray 자체를 저장
            현재는 유동장 데이터만 저장하게끔 함
    '''
    saveFlowDiff_NPY(flow_diff)
    '''
    Case 2. diff data file이 있을 때
        2-1. Load img file
        getDiff()는 위의 output_dir에 각 폴더에 차이 데이터 이미지가 저장되어있는 것을 전제로 함
        이는 초기에 getDiff(False)를 통해 각 폴더에 차이 데이터 이미지를 저장하는 것을 상정했기 때문임
    '''
    #flow_diff, design_diff = getDiff()
    '''
        2-2. Load npy file
            현재는 유동장 데이터만 저장했기 때문에, 유동장 데이터만 불러옴
    '''
    #flow_diff = loadFlowDiff_NPY()

#-------------------------------------------------------
# Update log
#-------------------------------------------------------
'''
2022.05.18
    - flowXX.csv, case.csv 등 주어진 csv 파일 구조 이해
    - 기존에 제작해주신 keras model을 pytorch로 이식 중, 문제점 발견, code를 갈아엎기로 결정
        문제점 : input size가 이상함
    - 이미 작성된 데이터 전처리 부분을 열심히 째려봤지만 너무 난해하여 이를 참고하지 않고 만들기로 결정
    - 주어진 preprocessing.py는 csv를 사용한 방식이 아니라서 직관적으로 이해하기 어려워 csv를 이용한 방식으로 정함
    - design factor가 저장되어있는 case.csv 정보를 불러오는 load_design() 메소드 작성
    - field factors가 저장되어있는 flowXX.csv 정보를 불러오는 load_fieldcsvList(), load_fieldcsv(), load_fieldcsvs() 메소드 작성
    - field 데이터를 csv파일별, factor(field)별로 보기 편하고 2d image형태로 볼 수 있게끔 numpy.ndarray([csv파일별, factor별, 높이(z)=402, 너비(y)=602]) 형식으로 제작
'''
'''
2022.05.19
    - field data가 (0,0)이 아닌 (0,602)에서 시작하는 것을 감안하여 np.flip을 추가해줌
    - design difference 정보를 저장해줄 csv파일 생성 메소드 작성
    - field difference 이미지를 만들고 저장해줄 메소드 작성
    - field image 및 difference image 오류 픽스 중
        field image 및 difference image를 생성하는 알고리즘의 문제가 아닌, flowXX.csv 파일의 문제인 것으로 파악됨
        flow01.csv와 flow02.csv만 정상적이며, 나머지는 같은 내용이 반복되는 등의 문제가 존재함
'''
'''
2022.05.20
    - 유동장 이미지를 토대로 brief하게 확인 결과 정상적인 데이터는 다음과 같음
        flow_01.csv
        flow_02.csv
        flow_07.csv
        flow_13.csv
        flow_18.csv
        flow_32.csv
        flow_34.csv
        flow_35.csv
        flow_46.csv
        flow_55.csv
        flow_61.csv
        이 중, flow_55.csv는 x,y,z 좌표 순서가 일정하지 않다는 문제점이 있기는 하나, 중복되는 좌표는 없어 데이터 손실은 없음

    - To do list
        - x, y, z 좌표가 중복되는 경우를 검출하는 코드 작성
            -> 정상 데이터를 확보하셨다고 하니 불필요함
        - 중복되지 않는다면 z 내림차순 -> y 내림차순 순으로 재배치 해주는 코드 작성(ex. flow_55.csv)
            -> 위와 동일
        - 이미지를 만들 때 normalization이 어떻게 되는지 확인 및 normalization 방법 고민 필요
'''
'''
2022.05.23
    - 유동장 데이터 및 차이 데이터를 cv.imwrite를 활용해 이미지로 만들었을 때의 특성은 다음과 같음
        [0 255] 범위의 정수값으로 데이터가 저장됨
        -> 소수점 아래의 정보는 반올림되어 정보 손실이 어느정도 발생함
        -> [0 255] 외의 범위는 정규화 과정을 거치며 정보가 왜곡될 수 있음
    - tiff(.tif파일)로 이미지를 저장하면 정보 손실은 없으나, 이미지 한장에 18~19GB정도의 크기로 저장됨
        -> 데이터 보관 및 처리에 있어 분명히 문제가 되는 부분임
'''
'''
2022.05.26
    - 실무진 회의 내용을 요약하면 다음과 같음
        디자인 인자의 차이는 선형적이나, 유동장 데이터의 차이는 선형적이지 않을 것이다.
        객체지향, 함수형 프로그램화를 통해 차후 다른 프로젝트에서도 사용할 수 있게 해주면 좋겠다.
            e.g. size가 달라도 적용할 수 있거나, size를 찾아내는 메소드를 추가하거나, 유류장의 class화
        #유동장 분석 노하우 : 어떤 변수를 넣어주는 것이 좋은가(송시몬 교수님 연구실)
        기본 정보로부터 다른 form의 데이터로 변환하는 것이 용이했으면 좋겠다.
        유동장에 noise를 섞어 augmentation 하는 것에 관한 논의
=======
''' 
mkDiff_csv.py
save u,v,w,vorticity differences and design differences

Maker info.
Name                : Yun ChanHyeok
Last update date    : 05/23/2022
'''
import numpy as np
import pandas as pd
import os
import cv2
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

# if __name__ == '__main__':
#     global base_dir, design_dir, field_dir, height, width, num_design, design_typeNum, output_dir
#     #global original_field_img_dir

#     # 데이터 위치 지정
#     base_dir = 'C:/Users/user/Desktop/code/hyundai/'
#     design_dir = base_dir + 'case.csv'
#     field_dir = base_dir + 'flowRearranged/'
#     output_dir = base_dir + 'dataset/'
#     height = 402
#     width = 602
#     num_design = 72
#     design_typeNum = 16
#     # 유동장 데이터 검토용
#     # original_field_img_dir = output_dir + 'ori_field_img/'
#     # os.makedirs(original_field_img_dir, exist_ok=True)

#     os.chdir(base_dir)
#     mkdirs()

#     '''
#     Case 1. diff data file이 없을 때
#         1-1. 유동장 차이 데이터는 image로 저장하고, 디자인 차이 데이터는 csv로 저장
#             * output * (getDiff)
#             flow_diff : 2556 x 4 x 402 x 602 ndarray
#             design_diff : Dataframe type
#                 1열 : idx1 : flow_diff_{}_{}의 형태로, {}번째 디자인과 {}번쨰 디자인을 비교한 경우인지를 표시.
#                 2열 : CD
#                 3열~17열 : 디자인요소
#     '''
#     flow_diff, design_diff = getDiff(False)
#     '''
#         1-2. 이미지 파일이 아닌 .npy파일로 ndarray 자체를 저장
#             현재는 유동장 데이터만 저장하게끔 함
#     '''
#     saveFlowDiff_NPY(flow_diff)
#     '''
#     Case 2. diff data file이 있을 때
#         2-1. Load img file
#         getDiff()는 위의 output_dir에 각 폴더에 차이 데이터 이미지가 저장되어있는 것을 전제로 함
#         이는 초기에 getDiff(False)를 통해 각 폴더에 차이 데이터 이미지를 저장하는 것을 상정했기 때문임
#     '''
#     #flow_diff, design_diff = getDiff()
#     '''
#         2-2. Load npy file
#             현재는 유동장 데이터만 저장했기 때문에, 유동장 데이터만 불러옴
#     '''
#     #flow_diff = loadFlowDiff_NPY()

#-------------------------------------------------------
# Update log
#-------------------------------------------------------
'''
2022.05.18
    - flowXX.csv, case.csv 등 주어진 csv 파일 구조 이해
    - 기존에 제작해주신 keras model을 pytorch로 이식 중, 문제점 발견, code를 갈아엎기로 결정
        문제점 : input size가 이상함
    - 이미 작성된 데이터 전처리 부분을 열심히 째려봤지만 너무 난해하여 이를 참고하지 않고 만들기로 결정
    - 주어진 preprocessing.py는 csv를 사용한 방식이 아니라서 직관적으로 이해하기 어려워 csv를 이용한 방식으로 정함
    - design factor가 저장되어있는 case.csv 정보를 불러오는 load_design() 메소드 작성
    - field factors가 저장되어있는 flowXX.csv 정보를 불러오는 load_fieldcsvList(), load_fieldcsv(), load_fieldcsvs() 메소드 작성
    - field 데이터를 csv파일별, factor(field)별로 보기 편하고 2d image형태로 볼 수 있게끔 numpy.ndarray([csv파일별, factor별, 높이(z)=402, 너비(y)=602]) 형식으로 제작
'''
'''
2022.05.19
    - field data가 (0,0)이 아닌 (0,602)에서 시작하는 것을 감안하여 np.flip을 추가해줌
    - design difference 정보를 저장해줄 csv파일 생성 메소드 작성
    - field difference 이미지를 만들고 저장해줄 메소드 작성
    - field image 및 difference image 오류 픽스 중
        field image 및 difference image를 생성하는 알고리즘의 문제가 아닌, flowXX.csv 파일의 문제인 것으로 파악됨
        flow01.csv와 flow02.csv만 정상적이며, 나머지는 같은 내용이 반복되는 등의 문제가 존재함
'''
'''
2022.05.20
    - 유동장 이미지를 토대로 brief하게 확인 결과 정상적인 데이터는 다음과 같음
        flow_01.csv
        flow_02.csv
        flow_07.csv
        flow_13.csv
        flow_18.csv
        flow_32.csv
        flow_34.csv
        flow_35.csv
        flow_46.csv
        flow_55.csv
        flow_61.csv
        이 중, flow_55.csv는 x,y,z 좌표 순서가 일정하지 않다는 문제점이 있기는 하나, 중복되는 좌표는 없어 데이터 손실은 없음

    - To do list
        - x, y, z 좌표가 중복되는 경우를 검출하는 코드 작성
            -> 정상 데이터를 확보하셨다고 하니 불필요함
        - 중복되지 않는다면 z 내림차순 -> y 내림차순 순으로 재배치 해주는 코드 작성(ex. flow_55.csv)
            -> 위와 동일
        - 이미지를 만들 때 normalization이 어떻게 되는지 확인 및 normalization 방법 고민 필요
'''
'''
2022.05.23
    - 유동장 데이터 및 차이 데이터를 cv.imwrite를 활용해 이미지로 만들었을 때의 특성은 다음과 같음
        [0 255] 범위의 정수값으로 데이터가 저장됨
        -> 소수점 아래의 정보는 반올림되어 정보 손실이 어느정도 발생함
        -> [0 255] 외의 범위는 정규화 과정을 거치며 정보가 왜곡될 수 있음
    - tiff(.tif파일)로 이미지를 저장하면 정보 손실은 없으나, 이미지 한장에 18~19GB정도의 크기로 저장됨
        -> 데이터 보관 및 처리에 있어 분명히 문제가 되는 부분임
'''
'''
2022.05.26
    - 실무진 회의 내용을 요약하면 다음과 같음
        디자인 인자의 차이는 선형적이나, 유동장 데이터의 차이는 선형적이지 않을 것이다.
        객체지향, 함수형 프로그램화를 통해 차후 다른 프로젝트에서도 사용할 수 있게 해주면 좋겠다.
            e.g. size가 달라도 적용할 수 있거나, size를 찾아내는 메소드를 추가하거나, 유류장의 class화
        #유동장 분석 노하우 : 어떤 변수를 넣어주는 것이 좋은가(송시몬 교수님 연구실)
        기본 정보로부터 다른 form의 데이터로 변환하는 것이 용이했으면 좋겠다.
        유동장에 noise를 섞어 augmentation 하는 것에 관한 논의
'''
'''
2022.05.27
    - ResNet18을 통해 학습 및 test해본 결과는 다음과 같음
        최고 Accuracy 13.281 %
        test loss가 불안정함
        가장 test loss가 안정적인 hyper parameter 조합에서 acc은 평균 8~9 %
    - 어떻게 조정식 연구원님은 높은 정확도를 얻을 수 있었는가?
        제공받은 코드 분석 결과, pd.get_dummies() 함수를 사용해 target label을 형성하는데,
        이는 모든 경우의 수를 다루는 것이 아닌 현존하는 경우의 수만을 target으로 함
        따라서 정답확률이 올라갈 수 밖에 없을 것으로 생각됨
        -> 이렇게 적용해보고 결과 확인볼 것.
>>>>>>> efb2b4ac1f288df262e55a9946ec71f22c0715cf
'''