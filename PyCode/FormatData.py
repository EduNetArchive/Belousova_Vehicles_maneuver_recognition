import csv
import numpy as np
import glob
import sys

class FormatData:
    def get_file_names_from_dir(fdir='..\\FlightImuData'):
        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            return glob.glob(fdir + "/*")
        else:
            return glob.glob(fdir + "\\*")
        
    def get_IMU_file_names_from_dir(fdir='..\\FlightImuData'):
        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            return glob.glob(fdir + "/IMU*")
        else:
            return glob.glob(fdir + "\\IMU*")
    
    def get_movie_file_names_from_dir(fdir='..\\FlightImuData'):
        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            return glob.glob(fdir + "/movie*")
        else:
            return glob.glob(fdir + "\\movie*")
        
    def get_files_dir():
        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            filepath = FormatData.get_IMU_file_names_from_dir(fdir=FormatData.labeled_files_dir_colab)
        else:
            filepath = FormatData.get_IMU_file_names_from_dir(fdir=FormatData.labeled_files_dir_local)
        return filepath
    
    '''
        For labeled data
    '''
    labeled_files_dir_local= '..\FlightImuDataLabeled'
    labeled_files_dir_colab= '/content/drive/MyDrive/2019-/NN/FANUC/FlightImuDataLabeled'
    
    def get_labeled_IMU_file_name(file_timestamp="2021_06_27_19_40_10"):
        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            return FormatData.labeled_files_dir_colab + '/IMU_' + file_timestamp + ".csv"
        else:
            return FormatData.labeled_files_dir_local + '\IMU_' + file_timestamp + ".csv"
        
    def get_labeled_files_dir():
        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            filepath = FormatData.labeled_files_dir_colab
        else:
            filepath = FormatData.labeled_files_dir_local
        return filepath
    
    '''
        For not labeled data
    '''
    files_dir_local = '..\FlightImuData'
    files_dir_colab = '/content/drive/MyDrive/2019-/NN/FANUC/FlightImuData'
    
    def get_IMU_file_name(file_timestamp="2021_06_27_19_40_10"):
        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            return FormatData.files_dir_colab + '/IMU_' + file_timestamp + ".csv"
        else:
            return FormatData.files_dir_local + '\IMU_' + file_timestamp + ".csv"
    
    def get_movie_file_name(file_timestamp="2021_06_27_19_40_10"):
        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            return FormatData.files_dir_colab + '/movie_' + file_timestamp + ".csv"
        else:
            return FormatData.files_dir_local + '\movie_' + file_timestamp + ".csv"
    
    '''
        Read and Write
    '''
    
    def read_data_from_csv(file_name="", b_print_end_msg = True):
        with open(file_name, "r", newline='\n') as csvfile:
            dataReader = csv.reader(
                csvfile,
                delimiter='\n',
                quotechar=' ')
            output = []
            for item in dataReader:
                output.append(item[0])
            csvfile.close()
            if b_print_end_msg:
                print("Data read!")
            return output
        
    def write_data_to_csv(file_name="", data_list=[], b_print_end_msg=True):
        with open(file_name, "w", newline='') as csvfile:
            dataWriter = csv.writer(
                csvfile,
                delimiter='\n',
                #quotechar=' ',
                #quoting = csv.QUOTE_NONNUMERIC
                )
            dataWriter.writerow(data_list)
            csvfile.close()
            if b_print_end_msg:
                print("Data saved!")
            
    '''
        Parse and Convert
    '''
    
    def parse_csv_data(data_list=[], legend_delimiter=', ', data_delimiter=',', has_legend=True):
        '''
            For movie data use legend_delimiter = ',', data_delimiter = ','
            For IMU data use legend_delimiter = ', ', data_delimiter = ','
        '''
        out_data_list = []
        
        if has_legend:
            legend = data_list[0].split(legend_delimiter)
            legLen = len(legend)
        else:
            legend = ""
            legLen = len(data_list[0].split(data_delimiter))
        
        for i in range(1 if has_legend else 0, len(data_list)):
            tmp = data_list[i].split(data_delimiter)
            if len(tmp) != legLen:
                print("Error at", i+1,"th line:", data_list[i])
            else:
                out_data_list.append(tmp)
        return legend, out_data_list
    
    def data_as_numpy_arrays(data_list=[]):
        data = []
        for i in range(len(data_list)):
            line = data_list[i]
            npline = np.zeros(len(line))
            for j in range(len(line)):
                npline[j] = float(line[j])
            data.append(npline)
        return np.array(data)
    
    def IMU_data_as_numpy_arrays(data_list=[]):
        '''
            Every IMU file has Timestamp[nanosec], 
                gx[rad/s], gy[rad/s], gz[rad/s], 
                ax[m/s^2], ay[m/s^2], az[m/s^2]
            Labeled file also has label.
            Outs: Timestamp[sec], g[rad/s], a[m/s^2], labels
        '''
        timestamp = []
        g = []
        a = []
        labels = []
        bHasLabeling = (len(data_list[0]) > 7)
        for i in range(len(data_list)):
            line = data_list[i]
            timestamp.append(line[0])# * 1e-9)
            g.append(np.array([float(line[1]), float(line[2]), float(line[3])]))
            a.append(np.array([float(line[4]), float(line[5]), float(line[6])]))
            if bHasLabeling:
                labels.append(int(line[7]))
        return np.array(timestamp).astype(np.longlong), np.array(g), np.array(a), np.array(labels)
    
    def movie_data_as_numpy_arrays(data_list=[]):
        '''
            Every Movie file has Timestamp[nanosec],
                fx[px],fy[px],Frame No.,Exposure time[nanosec],
                Sensor frame duration[nanosec],Frame readout time[nanosec],
                ISO,Focal length,Focus distance,AF mode
            Outs: Timestamp[sec]
        '''
        timestamp = []
        for i in range(len(data_list)):
            line = data_list[i]
            timestamp.append(float(line[0]))# * 1e-9)
        return np.array(timestamp)
    
    def numpy_arrays_as_IMU_string_data(timestamp, g: np.ndarray, a: np.ndarray, labels=[], legend=''):
        dataLines = []
        if legend:
            dataLines.append(legend)            
        bHasLabeling = (len(labels) > 0)
        for i in range(len(timestamp)):
            line = f"{timestamp[i]},{g[i,0]},{g[i,1]},{g[i,2]},{a[i,0]},{a[i,1]},{a[i,2]}"
            if bHasLabeling:
                line = line + f",{labels[i]}"
            dataLines.append(line)
        return dataLines
    
    def arrays_as_string_data(data, labels, dlegend='', b_remove_last=False):
        dataLines = []
        if dlegend:
            dataLines.append(dlegend)            
        bHasLabeling = (len(labels) > 0)
        # if bHasLabeling:
        #     if labels[0] > 5:
        #         print(labels[0:15])
        
        line_len = len(data[0])
        for i in range(len(data)):
            line = f"{data[i][0]}"
            for j in range(1, (line_len - 1) if b_remove_last else line_len):
                line = line + f",{data[i][j]}"
            if bHasLabeling:
                line = line + f",{labels[i]}"
            dataLines.append(line)
        return dataLines

    def to_categorical(y, num_classes):
        ''' 
        1-hot encodes a tensor 
        '''
        return np.eye(num_classes, dtype='uint8')[y]
    
    def make_data_split_for_parts(data, num_parts=10):
        out_data = [[] for i in range(10)]
        for i in range(len(data)):
            out_data[i%num_parts].append(data[i])
        return out_data