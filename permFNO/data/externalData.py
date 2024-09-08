import os
import h5py
import scipy.io
import numpy as np

def loadMatFile(path, entry):
    try:
        f = scipy.io.loadmat(path)
        print(f)
        print('raw')
        return np.array(f[entry])

    except NotImplementedError:
        with h5py.File(path, 'r') as f:
            print('compressed')
            data = np.array(f[entry])
            data = np.swapaxes(data, 0, 2)
            return data

    except:
        ValueError('could not open the file :(')

#paths
project_path = '/home/woody/iwia/iwia057h/external/srv/www/digrocks/portal/media/projects/372/'
origin_path = 'origin/'
analysis_path = 'analysis'
simulation_type = 'P_5_MPa' + '.mat'

path = project_path + origin_path

input = np.zeros((150, 256, 256, 256))
#rho = np.zeros((150, 256, 256, 256))
#vz = np.zeros((150, 256, 256, 256))
exclude = []
names = []
#PROBLEM reading file:  344_05_256.mat
pos = 0
for i in range(1999, 2300):#2205):
    folder = os.path.join(path, str(i), 'images')
    for root, dirs, files in os.walk(folder, topdown=False):
        for file in files:

            #analysis = os.path.join(project_path, analysis_path, str(i-1239), 'images', simulation_type)
            if file.endswith('256.mat'):
                #if os.path.exists(analysis):

                print('reading file: ', file)

                #get geometry data
                origin = os.path.join(root, file)
                input[pos] = loadMatFile(origin, 'bin')

                #get simulation data
                #rho[pos] = loadMatFile(analysis, 'rho')
                #vz[pos] = loadMatFile(analysis, 'uz')

                #add name
                names.append(file)
                pos += 1

                #else:
                    #print('no analytical solution: ', file)
                    #exclude.append(pos)

print("count: ", pos)

inputs = {}
inputs['fill'] = input[0:pos]

outputs = {}
#outputs['p'] = rho[0:pos]
#outputs['z'] = vz[0:pos]


def saveH5PY(input_data, output_data, name_data, filename):
    with h5py.File(filename, "w") as f:
        input = f.create_group("input")
        for key, value in input_data.items():
            input.create_dataset(key, data=value)

        #output = f.create_group("output")
        #for key, value in output_data.items():
            #output.create_dataset(key, data=value)

        asciiList = [n.encode("ascii", "ignore") for n in name_data]
        f.create_dataset("name", data=asciiList)

saveH5PY(inputs, outputs, names, '/home/woody/iwia/iwia057h/external/external2.hp5')

