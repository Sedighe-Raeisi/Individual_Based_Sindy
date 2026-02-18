import numpy as np

class gen_param:
    def __init__(self, N_param, a_V = None, a_mean = None, a_std = None, zero_peak_portion = None, zero_epsilon = 0.001,
                 secondpeak_posrtion = None, secondpeak_mean = None, secondpeak_std = None):
        self.N_param = N_param
        self.a_V = a_V
        self.a_mean = a_mean
        self.a_std = a_std
        self.zero_portion = zero_peak_portion
        self.gen_count = 0
        self.zero_epsilon = zero_epsilon
        self.secondpeak_posrtion = secondpeak_posrtion
        self.secondpeak_mean = secondpeak_mean
        self.secondpeak_std = secondpeak_std


    def gen(self):
        if self.a_V is not None:
            return self.a_V
        elif self.a_mean is not None and self.a_std is not None:


            if self.secondpeak_posrtion is not None:
                print("############################# second peak activated ######################")
                secondpeak_N = int(self.secondpeak_posrtion * self.N_param)
                if self.gen_count<secondpeak_N:
                    self.gen_count+=1
                    return np.random.normal(self.secondpeak_mean, self.secondpeak_std)
                else:
                    print("second_portion_cont = " ,self.gen_count)
                    self.gen_count+= 1
                    return np.random.normal(self.a_mean, self.a_std)


            elif self.zero_portion is not None:
                a_N_zero = int(self.zero_portion * self.N_param)
                if self.gen_count<a_N_zero:
                    self.gen_count+=1
                    return np.random.normal(0, self.zero_epsilon)
                else:
                    self.gen_count+= 1
                    return np.random.normal(self.a_mean, self.a_std)

            # elif self.zero_portion is None and self.secondpeak_posrtion is None:
            else:
                self.gen_count += 1
                return np.random.normal(self.a_mean, self.a_std)


        else:
            print(f"Error! mean = {self.a_mean} or std = {self.a_std} not found")
